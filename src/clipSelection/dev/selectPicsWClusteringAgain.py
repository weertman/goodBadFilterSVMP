# selectPicsWClustersAgain.py

import os
import sys
import logging
import random
import shutil
import traceback
import gc
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
import torch
import matplotlib

matplotlib.use('Agg')  # for headless environments

import clip
from interactivePlots import create_interactive_umap_plot
from checkDuplicates import check_duplicates  # Ensure checkDuplicates.py is on your Python path

# ----------------------------
# Configuration
# ----------------------------
RUN_NUM = 1  # e.g. 1, 2, 3, etc.

# Directory with "already chosen" images you want to exclude duplicates from
chosen_dir = Path('/Users/wlweert/Documents/python/toolsToHandlePhotos/src/clustering/SORTED_STARS_CROPPED_clusters')

# The "pool" of images to cluster
path_root_dir = Path('/Users/wlweert/Documents/python/toolsToHandlePhotos/src/sunflowerStarArchive/SORTED_STARS_CROPPED')
path_dst_dir = Path(f'{path_root_dir.name}_clusters')
path_dst_dir.mkdir(exist_ok=True)

N_images = 500         # Target number of images to sample evenly across clusters
num_clusters = 20      # Number of clusters
BATCH_SIZE = 64        # Batch size for embedding
NUM_PCA_COMPONENTS = 15
RANDOM_SEED = 42
checkpoint_file = path_dst_dir / "embeddings_checkpoint.npz"

# ----------------------------
# Logging Setup
# ----------------------------
logger = logging.getLogger('image_clustering')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
handler.setFormatter(formatter)

# Avoid adding multiple handlers in interactive sessions:
if not logger.handlers:
    logger.addHandler(handler)

# ----------------------------
# Utility Functions
# ----------------------------
def find_images(root_dir: Path):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    return [p for p in root_dir.rglob('*') if p.suffix.lower() in valid_extensions]

def load_image(image_path: Path, preprocess=None):
    try:
        with Image.open(image_path).convert('RGB') as img:
            if preprocess:
                return preprocess(img)
            return img
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None

def extract_embedding(model, device, image_batch):
    image_batch = image_batch.to(device, non_blocking=True)
    with torch.no_grad():
        emb = model.encode_image(image_batch)
        # Normalize embeddings
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

def initialize_model(device):
    # Load the CLIP model and its preprocess function
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess

def load_and_embed_images(image_paths, model, device, preprocess, batch_size=BATCH_SIZE):
    """
    Convert a list of image paths into a list of embeddings using CLIP.
    Returns (valid_image_paths, embeddings).
    """
    embeddings = []
    valid_image_paths = []

    for start_idx in tqdm(range(0, len(image_paths), batch_size),
                          desc='Embedding images', unit='batch'):
        batch_paths = image_paths[start_idx:start_idx + batch_size]
        batch_images = []
        for pth in batch_paths:
            img = load_image(pth, preprocess=preprocess)
            if img is not None:
                batch_images.append(img)
                valid_image_paths.append(pth)

        if batch_images:
            batch_tensor = torch.stack(batch_images, dim=0)
            batch_embeddings = extract_embedding(model, device, batch_tensor)
            embeddings.append(batch_embeddings)

        # Free memory
        del batch_images
        if 'batch_tensor' in locals():
            del batch_tensor

        torch.cuda.empty_cache()
        gc.collect()

    if embeddings:
        embeddings = np.vstack(embeddings)
    return valid_image_paths, embeddings

def sample_images_across_clusters(cluster_to_images, N_images):
    """
    Randomly sample N_images by iterating over clusters in random order,
    picking one image per cluster per round,
    until we either reach N_images or no cluster has images left.
    """
    all_clusters = list(cluster_to_images.keys())
    sampled_images = []

    # Continue until we have enough images or no clusters have images left
    while len(sampled_images) < N_images and any(len(cluster_to_images[c]) > 0 for c in all_clusters):
        random.shuffle(all_clusters)
        for c in all_clusters:
            if len(sampled_images) >= N_images:
                break
            if cluster_to_images[c]:
                chosen_img = random.choice(cluster_to_images[c])
                cluster_to_images[c].remove(chosen_img)
                sampled_images.append((chosen_img, c))
                if len(sampled_images) >= N_images:
                    break

    return sampled_images

# ----------------------------
# Main Script
# ----------------------------
if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    try:
        logger.info("Gathering images from the pool directory...")
        all_image_paths = find_images(path_root_dir)
        if not all_image_paths:
            logger.error("No images found in the pool directory.")
            sys.exit(1)

        logger.info("Excluding duplicates based on chosen_dir images...")
        duplicates = check_duplicates(
            chosen_dir=chosen_dir,
            pool_dir=path_root_dir,
            threshold=0.995,  # Adjust as desired
            batch_size=BATCH_SIZE,
            num_threads=4
        )

        old_count = len(all_image_paths)
        all_image_paths = [p for (p, is_dup) in zip(all_image_paths, duplicates) if not is_dup]
        new_count = len(all_image_paths)
        logger.info(f"Filtered out {old_count - new_count} duplicates. Remaining pool size: {new_count}")

        if not all_image_paths:
            logger.error("All pool images were flagged as duplicates; nothing left to cluster.")
            sys.exit(0)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Overwrite the old checkpoint file, to ensure we only embed these images
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if checkpoint_file.exists():
            logger.info(f"Removing old checkpoint file: {checkpoint_file}")
            checkpoint_file.unlink()

        logger.info("Extracting embeddings from scratch (overwriting checkpoint)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess = initialize_model(device)
        valid_image_paths, embeddings = load_and_embed_images(all_image_paths, model, device, preprocess)

        # Save the new checkpoint
        np.savez_compressed(checkpoint_file, image_paths=valid_image_paths, embeddings=embeddings)

        # Free the model and GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Dimensionality reduction
        num_samples = embeddings.shape[0]
        if num_samples < NUM_PCA_COMPONENTS:
            logger.warning(
                f"Only {num_samples} samples present. "
                f"Reducing PCA components from {NUM_PCA_COMPONENTS} to {num_samples}"
            )
            NUM_PCA_COMPONENTS = num_samples

        logger.info("Performing IncrementalPCA dimension reduction...")
        ipca = IncrementalPCA(n_components=NUM_PCA_COMPONENTS, batch_size=256)

        first_batch_size = max(NUM_PCA_COMPONENTS, 256)
        if first_batch_size > num_samples:
            first_batch_size = num_samples
            if first_batch_size < NUM_PCA_COMPONENTS:
                logger.warning(
                    f"First batch size {first_batch_size} < {NUM_PCA_COMPONENTS} components; "
                    f"Reducing to {first_batch_size}."
                )
                NUM_PCA_COMPONENTS = first_batch_size
                ipca = IncrementalPCA(n_components=NUM_PCA_COMPONENTS, batch_size=256)

        ipca.partial_fit(embeddings[:first_batch_size])

        chunk_size = 256
        for start in range(first_batch_size, num_samples, chunk_size):
            end = start + chunk_size
            ipca.partial_fit(embeddings[start:end])

        # Transform embeddings
        embeddings_pca_list = []
        for start in range(0, num_samples, chunk_size):
            end = start + chunk_size
            chunk_transformed = ipca.transform(embeddings[start:end])
            embeddings_pca_list.append(chunk_transformed)
        embeddings_pca = np.vstack(embeddings_pca_list)

        del embeddings
        gc.collect()
        torch.cuda.empty_cache()

        # Cluster with Spectral Clustering
        logger.info("Clustering images...")
        knn_graph = kneighbors_graph(embeddings_pca, n_neighbors=15, include_self=False)
        clustering = SpectralClustering(
            n_clusters=num_clusters,
            affinity='precomputed',
            random_state=RANDOM_SEED
        )
        cluster_labels = clustering.fit_predict(knn_graph)

        logger.info("Grouping images by cluster...")
        cluster_to_images = {c: [] for c in range(num_clusters)}
        for path, label in zip(valid_image_paths, cluster_labels):
            cluster_to_images[label].append(path)

        logger.info("Sampling images across clusters...")
        sampled_images = sample_images_across_clusters(cluster_to_images, N_images)
        if len(sampled_images) < N_images:
            logger.warning(f"Only {len(sampled_images)} images sampled out of {N_images} requested.")

        # Remove old cluster folders and create new ones
        logger.info("Removing old cluster folders (if any) and creating new ones with run_num tags...")
        for c in range(num_clusters):
            cluster_dir = path_dst_dir / f"{RUN_NUM}_cluster_{c}"
            if cluster_dir.exists():
                shutil.rmtree(cluster_dir)
            cluster_dir.mkdir(parents=True, exist_ok=False)

        logger.info("Copying sampled images to run_num cluster folders...")
        for img_path, c in tqdm(sampled_images, desc="Copying images"):
            cluster_dir = path_dst_dir / f"{RUN_NUM}_cluster_{c}"
            dst_file = cluster_dir / img_path.name
            shutil.copy(img_path, dst_file)

        logger.info("Generating UMAP visualization...")
        sampled_indices = [valid_image_paths.index(img_path) for img_path, _ in sampled_images]
        create_interactive_umap_plot(
            valid_image_paths,
            embeddings_pca,
            dimension='2d',
            cluster_labels=cluster_labels,
            highlight_indices=set(sampled_indices),
            output_file=path_dst_dir / f'umap_plot_run_{RUN_NUM}.html'
        )

        del embeddings_pca
        del cluster_labels
        del cluster_to_images
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Process completed successfully for run_num={RUN_NUM}.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
