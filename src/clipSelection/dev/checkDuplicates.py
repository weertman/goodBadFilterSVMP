# checkDuplicates.py

import os
import sys
import math
import time
import json
import random
import logging
from pathlib import Path
from typing import List, Optional

import torch
import clip
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# -----------------------------------------------------
# Logging Setup (minimal; adjust to your needs)
# -----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # or INFO as needed
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def _valid_image_extensions():
    """Return a set of valid image extensions for scanning directories."""
    return {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}


def _find_images_in_dir(directory: Path) -> List[Path]:
    """
    Recursively find all image files under the given directory.
    Returns a sorted list of Paths.
    """
    if not directory.exists():
        logger.warning(f"Directory '{directory}' does not exist.")
        return []
    all_files = list(directory.rglob('*'))
    valid_exts = _valid_image_extensions()
    image_paths = [f for f in all_files if f.suffix.lower() in valid_exts]
    return sorted(image_paths)


def _load_one_image(img_path: Path, preprocess) -> torch.Tensor:
    """
    Helper function to load + preprocess a single image with PIL+CLIP.
    Returns a 3D or 4D tensor (usually 3x224x224).
    """
    try:
        with Image.open(img_path).convert('RGB') as img:
            return preprocess(img)
    except Exception as e:
        logger.error(f"Failed to load image {img_path}: {e}")
        return None


def _parallel_load_images(
    image_paths: List[Path],
    preprocess,
    num_threads: int = 4
) -> List[Optional[torch.Tensor]]:
    """
    Load images in parallel using a ThreadPoolExecutor.
    Returns a list of Tensors (or None where loading failed),
    preserving the order of `image_paths`.
    """
    loaded_images = [None] * len(image_paths)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit load tasks
        futures = {}
        for i, p in enumerate(image_paths):
            futures[executor.submit(_load_one_image, p, preprocess)] = i

        # Gather results in order, with a tqdm progress bar
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Loading images (parallel)", unit="img"):
            i = futures[future]
            try:
                loaded_images[i] = future.result()
            except Exception as e:
                logger.error(f"Failed in future for {image_paths[i]}: {e}")
                loaded_images[i] = None

    return loaded_images


def _embed_images(
    image_paths: List[Path],
    model,
    preprocess,
    device,
    batch_size: int,
    num_threads: int
) -> (List[Path], np.ndarray):
    """
    Orchestrates:
      1) parallel load images (in threads)
      2) embed them in batches on the GPU
    Returns:
      valid_paths (list of Path) - only those images that successfully loaded
      embeddings (np.ndarray) - shape [N, D] of CLIP embeddings
    """
    logger.info(f"Parallel-loading {len(image_paths)} images with {num_threads} threads...")
    loaded_images = _parallel_load_images(image_paths, preprocess, num_threads)

    # Filter out failures
    valid_items = [(path, img_tensor)
                   for (path, img_tensor) in zip(image_paths, loaded_images)
                   if img_tensor is not None]
    if not valid_items:
        return [], np.empty((0, 512))

    # Sort back (already in correct order, but we want to get separate lists)
    valid_paths = [v[0] for v in valid_items]
    valid_tensors = [v[1] for v in valid_items]

    logger.info(f"Embedding {len(valid_items)} loaded images on device={device}...")
    all_embs = []

    # GPU feed in batches
    for start_idx in tqdm(range(0, len(valid_tensors), batch_size),
                          desc="Embedding batches", unit="batch"):
        batch_end = start_idx + batch_size
        batch_tensors = valid_tensors[start_idx:batch_end]
        batch_stack = torch.stack(batch_tensors).to(device, non_blocking=True)

        with torch.no_grad():
            emb = model.encode_image(batch_stack)
            # Normalize embeddings
            emb = emb / emb.norm(dim=-1, keepdim=True)

        all_embs.append(emb.cpu().numpy())

        del batch_stack, emb
        torch.cuda.empty_cache()

    if all_embs:
        embs_np = np.vstack(all_embs)
    else:
        embs_np = np.empty((0, 512))

    return valid_paths, embs_np


def _compute_cosine_sim_matrix(
    pool_embs: np.ndarray,
    chosen_embs: np.ndarray
) -> np.ndarray:
    """
    Compute the NxM cosine similarity matrix of pool_embs (N rows)
    vs. chosen_embs (M rows). Because both are L2-normalized, we can
    do a dot product. The result is shape (N, M).
    """
    # shape: [N, D] @ [D, M] -> [N, M]
    return pool_embs @ chosen_embs.T


def check_duplicates(
    chosen_dir: Path,
    pool_dir: Path,
    threshold: float = 0.9,
    batch_size: int = 32,
    num_threads: int = 4
) -> List[bool]:
    """
    Compare images in pool_dir vs. chosen_dir using CLIP embeddings.
    Returns a list of booleans (same length/order as the pool_dir’s image list),
    indicating which pool images are duplicates (should be excluded).

    Steps:
    1) Scan chosen_dir for images, embed them.
    2) Scan pool_dir for images, embed them.
    3) For each pool embedding, check if it has a similarity >= threshold
       to any chosen embedding. If so, mark exclude=True.

    Args:
      chosen_dir:    Path to folder with "already chosen" images.
      pool_dir:      Path to folder with images to check for duplicates.
      threshold:     Cosine similarity threshold. [0..1].
      batch_size:    Batch size for GPU embedding.
      num_threads:   Number of CPU threads to parallelize image loading.

    Returns:
      excludes: list of bool of length = number of pool images (in sorted order).
               True means "exclude / is duplicate."
    """
    # 1) Find chosen images
    chosen_paths = _find_images_in_dir(chosen_dir)
    logger.info(f"Found {len(chosen_paths)} chosen images in {chosen_dir}")
    # 2) Find pool images
    pool_paths = _find_images_in_dir(pool_dir)
    logger.info(f"Found {len(pool_paths)} pool images in {pool_dir}")

    # Edge cases
    if len(chosen_paths) == 0:
        logger.warning("No chosen images found. No duplicates will be flagged.")
        return [False] * len(pool_paths)
    if len(pool_paths) == 0:
        logger.warning("No pool images found. Returning empty list.")
        return []

    # 3) Load CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 4) Embed chosen images
    chosen_valid_paths, chosen_embs = _embed_images(
        chosen_paths, model, preprocess, device, batch_size, num_threads
    )
    logger.info(f"Embedded {len(chosen_valid_paths)}/{len(chosen_paths)} chosen images successfully.")

    if len(chosen_embs) == 0:
        logger.warning("All chosen images failed embedding or none found; no duplicates flagged.")
        return [False] * len(pool_paths)

    # 5) Embed pool images
    pool_valid_paths, pool_embs = _embed_images(
        pool_paths, model, preprocess, device, batch_size, num_threads
    )
    logger.info(f"Embedded {len(pool_valid_paths)}/{len(pool_paths)} pool images successfully.")

    # Now, we must keep track of which pool_paths correspond to valid embeddings
    # so we can return a bool list for *all* pool images in the original sorted order.
    # We'll do it by building an array of "exclusion flags" for valid items,
    # then re-insert them into the master list in correct positions.

    excludes_full = [False] * len(pool_paths)  # initialize

    if len(pool_embs) == 0:
        logger.warning("All pool images failed embedding or none found; returning all False.")
        return excludes_full  # no duplicates because no images

    # 6) Compute NxM similarity matrix
    logger.info("Computing similarity matrix between pool and chosen images...")
    sim_matrix = _compute_cosine_sim_matrix(pool_embs, chosen_embs)
    # shape is [N_pool, M_chosen]

    # 7) For each pool embedding row, find the maximum similarity to chosen
    max_sims = sim_matrix.max(axis=1)  # shape [N_pool,]
    duplicates_flags = (max_sims >= threshold)  # True/False array

    # 8) Map back to excludes_full
    # We know pool_valid_paths are a subset of pool_paths in the same order.
    pool_idx_map = {}
    valid_idx_counter = 0
    for i, ppath in enumerate(pool_paths):
        if valid_idx_counter < len(pool_valid_paths) and ppath == pool_valid_paths[valid_idx_counter]:
            # this means i-th in the full list was also i-th valid embedding
            pool_idx_map[i] = valid_idx_counter
            valid_idx_counter += 1
        else:
            pool_idx_map[i] = None

    for i in range(len(pool_paths)):
        v_idx = pool_idx_map[i]
        if v_idx is not None:
            # This pool image had an embedding
            excludes_full[i] = bool(duplicates_flags[v_idx])
        else:
            # This image had no embedding -> default False
            excludes_full[i] = False

    logger.info("Done checking duplicates. Summary:")
    total_excluded = sum(excludes_full)
    logger.info(f"Out of {len(pool_paths)} pool images, {total_excluded} are flagged as duplicates.")

    return excludes_full


# If you want to test quickly via command line, you could do:
# if __name__ == "__main__":
#     from argparse import ArgumentParser
#     ap = ArgumentParser()
#     ap.add_argument("--chosen", type=str, required=True)
#     ap.add_argument("--pool", type=str, required=True)
#     ap.add_argument("--threshold", type=float, default=0.9)
#     ap.add_argument("--batch-size", type=int, default=32)
#     ap.add_argument("--num-threads", type=int, default=4)
#     args = ap.parse_args()
#
#     chosen_dir = Path(args.chosen)
#     pool_dir = Path(args.pool)
#     excludes = check_duplicates(
#         chosen_dir=chosen_dir,
#         pool_dir=pool_dir,
#         threshold=args.threshold,
#         batch_size=args.batch_size,
#         num_threads=args.num_threads
#     )
#     print("Exclude flags:", excludes)
