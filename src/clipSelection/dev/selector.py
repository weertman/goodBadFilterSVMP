"""
configuration_rolling_loaders.py

Single-process GPU inference with multiple loader threads:
 - Each loader thread reads frames for a batch of clips, flattens & stacks them on CPU.
 - The main thread pulls these big CPU tensors from a queue, does .to(device) + encode_image().
 - Rest of your pipeline (PCA, UMAP, clustering, saving) is unchanged.
"""

from pathlib import Path
from tqdm import tqdm
import cv2
from datetime import datetime
from functools import partial
import threading
import queue
import multiprocessing as mp
import numpy as np
import torch
import random
from PIL import Image
import clip
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
import umap
import gc

############################################
# from visualizations import (
#     visualize_video_frame_lengths,
#     visualize_video_time_length_in_mins,
#     visualize_video_size_distribution,
# )
############################################

###########################################################
# 1) Metadata + Candidate Clip Functions (unchanged)
###########################################################
def open_video_get_fps_frame_count_and_size(path_video):
    cap = cv2.VideoCapture(str(path_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    cap.release()
    return path_video, fps, frame_count, size


def generate_candidate_clips_for_video(args, frames_per_clip=10, overlap=0.5):
    path_video, meta = args
    fps = meta["fps"]
    frame_count = meta["frame_count"]

    clip_length = frames_per_clip
    step = int(clip_length * (1.0 - overlap))

    candidate_clips = []
    start = 0
    while start < frame_count - clip_length:
        clip_info = {
            "video_path": path_video,
            "start_frame": start,
            "end_frame": start + clip_length,
            "fps": fps,
            "duration_seconds": clip_length / fps,
        }
        candidate_clips.append(clip_info)
        start += step

    return candidate_clips


def generate_candidate_clips(dict_subdir_to_video_metas, frames_per_clip=10, overlap=0.5, n_workers=4):
    video_meta_list = []
    for subdir, videos_dict in dict_subdir_to_video_metas.items():
        for path_video, meta in videos_dict.items():
            video_meta_list.append((path_video, meta))

    worker_fn = partial(
        generate_candidate_clips_for_video,
        frames_per_clip=frames_per_clip,
        overlap=overlap,
    )
    candidate_clips = []
    with mp.Pool(n_workers) as pool:
        results = pool.imap_unordered(worker_fn, video_meta_list)
        for clip_list in tqdm(results, total=len(video_meta_list), desc="Generating Clips"):
            candidate_clips.extend(clip_list)

    return candidate_clips


def split_random_vs_cluster(candidate_clips, ratio_random=0.5):
    random.shuffle(candidate_clips)
    n_random = int(ratio_random * len(candidate_clips))
    random_pool = candidate_clips[:n_random]
    cluster_pool = candidate_clips[n_random:]
    return random_pool, cluster_pool


def sample_random_clips_by_time(random_pool, target_minutes):
    random.shuffle(random_pool)
    selected = []
    total_s = 0.0
    target_s = target_minutes * 60.0
    for clip in random_pool:
        clip_dur = clip["duration_seconds"]
        if total_s + clip_dur <= target_s:
            selected.append(clip)
            total_s += clip_dur
        else:
            break
    return selected


def sample_clips_across_clusters_by_time(valid_clips, cluster_labels, target_minutes=30.0):
    from collections import defaultdict
    cluster_to_indices = defaultdict(list)
    for i, c in enumerate(cluster_labels):
        cluster_to_indices[c].append(i)

    for c in cluster_to_indices:
        random.shuffle(cluster_to_indices[c])

    cluster_ids = list(cluster_to_indices.keys())
    selected = []
    total_s = 0.0
    target_s = target_minutes * 60.0

    while total_s < target_s and any(cluster_to_indices[c] for c in cluster_ids):
        random.shuffle(cluster_ids)
        for c in cluster_ids:
            if total_s >= target_s:
                break
            if cluster_to_indices[c]:
                idx = cluster_to_indices[c].pop()
                clip_info = valid_clips[idx]
                dur = clip_info["duration_seconds"]
                if total_s + dur <= target_s:
                    selected.append((clip_info, c))
                    total_s += dur
                else:
                    pass
    return selected

###########################################################
# 2) Rolling Loader with Multiple Threads
###########################################################
def chunkify(lst, chunk_size):
    """
    Yield consecutive chunks of 'lst' up to chunk_size each.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

def read_frames_cpu_and_stack(clips_batch, preprocess_fn):
    """
    Reads frames for each clip in 'clips_batch', flattens them,
    and returns (clips_batch, big_stack_cpu, clip_frame_counts).

    big_stack_cpu shape = [N_total_frames, 3, H, W] on CPU
    clip_frame_counts: array of how many frames each clip has
    """
    all_frames = []
    clip_frame_counts = []
    for clip_info in clips_batch:
        frames_cpu = read_frames_for_one_clip(clip_info, preprocess_fn)
        clip_frame_counts.append(len(frames_cpu))
        all_frames.extend(frames_cpu)

    if not all_frames:
        # means no frames read at all
        return (clips_batch, None, clip_frame_counts)

    # stack on CPU
    big_stack_cpu = torch.stack(all_frames, dim=0)  # shape=(N, 3, H, W) on CPU
    return (clips_batch, big_stack_cpu, clip_frame_counts)


def read_frames_for_one_clip(clip_info, preprocess_fn):
    """
    Read frames from disk for one clip, apply CPU-based transform,
    return list of CPU Tensors (one per frame).
    """
    path_video = clip_info["video_path"]
    start_frame = clip_info["start_frame"]
    end_frame = clip_info["end_frame"]

    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        return []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    n_frames = end_frame - start_frame
    frames_cpu = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor_cpu = preprocess_fn(pil_img)  # shape=(3,H,W) on CPU
        frames_cpu.append(tensor_cpu)

    cap.release()
    return frames_cpu


def embed_big_batch_on_gpu(clips_batch, big_stack_cpu, clip_frame_counts, model, device):
    """
    Move big_stack_cpu to GPU, run model.encode_image, average per clip.
    Returns list of (clip_info, embedding).
    """
    # if big_stack_cpu is None => no frames
    if big_stack_cpu is None:
        # produce (clip_info, None) for all
        return [(clip_info, None) for clip_info in clips_batch]

    big_stack_gpu = big_stack_cpu.to(device)  # shape=(N,3,H,W)
    with torch.no_grad():
        emb = model.encode_image(big_stack_gpu)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    big_stack_gpu = None
    gc.collect()

    results = []
    idx_start = 0
    for clip_info, frame_count in zip(clips_batch, clip_frame_counts):
        if frame_count == 0:
            # no frames for this clip
            results.append((clip_info, None))
        else:
            this_emb = emb[idx_start : idx_start + frame_count]
            idx_start += frame_count
            clip_embedding = this_emb.mean(dim=0).cpu().numpy()
            results.append((clip_info, clip_embedding))

    del emb
    gc.collect()
    return results


def embed_all_clips_rolling_loader(candidate_clips, clips_per_batch=64, n_loader_threads=2):
    """
    Single-process GPU inference with multiple loader threads:
      - We chunk candidate_clips into groups of size 'clips_per_batch'
      - Each loader thread reads frames for these groups, flattens & stacks on CPU
      - Main thread pulls them from a queue, does .to(device) + encode_image in big batches
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess_fn = clip.load("ViT-L/14", device=device)
    model.eval()

    # chunk the clips
    batched_clips = list(chunkify(candidate_clips, clips_per_batch))
    num_batches = len(batched_clips)

    # We'll have a queue to store (clips_batch, big_stack_cpu, clip_frame_counts)
    # from loader threads. We'll do an easy approach: each thread loops over
    # certain batches. If you want dynamic scheduling, you'd put batch IDs in a job queue.
    data_queue = queue.Queue(maxsize=4)  # up to 4 large CPU Tensors in memory

    # We'll store the final results
    all_results = []

    # 1) Distribute the 'batched_clips' across n_loader_threads
    # e.g. if we have 10 batches, 2 threads => thread0 gets [0,2,4,6,8], thread1 gets [1,3,5,7,9]
    # or we can do contiguous splits
    def get_worker_batches(worker_id):
        # simple round-robin approach
        return [batched_clips[i] for i in range(worker_id, num_batches, n_loader_threads)]

    # 2) Loader thread function
    def loader_thread_func(worker_id):
        my_batches = get_worker_batches(worker_id)
        for clips_batch in my_batches:
            # read & stack
            triple = read_frames_cpu_and_stack(clips_batch, preprocess_fn)
            data_queue.put(triple)
        # when done, each thread puts a sentinel
        data_queue.put(None)

    # 3) Launch loader threads
    loaders = []
    for worker_id in range(n_loader_threads):
        t = threading.Thread(target=loader_thread_func, args=(worker_id,), daemon=True)
        t.start()
        loaders.append(t)

    # 4) Main thread: read from queue num_batches times *per thread*, but we have multiple sentinels
    total_sentinels_received = 0

    # we expect 'num_batches' items total, but they come from multiple threads in random order
    # each thread will produce len(get_worker_batches(worker_id)) items + 1 sentinel
    # total items = num_batches + n_loader_threads
    # We'll stop once we've processed 'num_batches' real items
    real_items_processed = 0

    pbar = tqdm(total=num_batches, desc="Embedding Batches (rolling)")

    while real_items_processed < num_batches:
        triple = data_queue.get()
        if triple is None:
            total_sentinels_received += 1
            if total_sentinels_received == n_loader_threads:
                # we've got all data
                break
            continue

        (clips_batch, big_stack_cpu, clip_frame_counts) = triple
        batch_res = embed_big_batch_on_gpu(clips_batch, big_stack_cpu, clip_frame_counts, model, device)
        all_results.extend(batch_res)
        real_items_processed += 1
        pbar.update(1)

    pbar.close()

    # ensure threads are done
    for t in loaders:
        t.join()

    return all_results

###########################################################
# 3) Dimensionality Reduction, Clustering, etc. (unchanged)
###########################################################
def do_incremental_pca(embeddings, n_components=64, batch_size=4096):
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    n_samples = embeddings.shape[0]
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        ipca.partial_fit(embeddings[start_idx:end_idx])

    pca_list = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        pca_list.append(ipca.transform(embeddings[start_idx:end_idx]))
    pca_embeddings = np.vstack(pca_list)
    return ipca, pca_embeddings


def do_umap(pca_embeddings, n_components=2, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=42,
    )
    umap_embeddings = reducer.fit_transform(pca_embeddings)
    return reducer, umap_embeddings


def cluster_with_spectral(umap_embeddings, n_clusters=250, n_neighbors=15, random_seed=42):
    adjacency = kneighbors_graph(
        umap_embeddings, n_neighbors=n_neighbors, include_self=False, n_jobs=-1
    )
    clustering = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=random_seed
    )
    cluster_labels = clustering.fit_predict(adjacency)
    return cluster_labels


###########################################################
# 4) Saving (unchanged)
###########################################################
def save_one_clip(args):
    clip_info, cluster_label, base_output_dir, idx = args
    video_path = clip_info["video_path"]
    start_frame = clip_info["start_frame"]
    end_frame = clip_info["end_frame"]
    fps = clip_info["fps"]

    subdir_name = f"clip_{idx:05d}"
    if cluster_label is not None:
        subdir_name += f"_cluster_{cluster_label}"
    output_subdir = base_output_dir / subdir_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    out_video_path = output_subdir / "clip.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (idx, False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return (idx, False)
    height, width = frame.shape[:2]

    # Rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    total_frames = end_frame - start_frame
    written_count = 0
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out_writer.write(frame)
        written_count += 1

    out_writer.release()
    cap.release()
    success = (written_count > 0)
    return (idx, success)


def save_selected_clips_multiprocessing(tasks, n_workers=4):
    results = []
    with mp.Pool(n_workers) as pool:
        for res in tqdm(pool.imap_unordered(save_one_clip, tasks), total=len(tasks), desc="Saving Clips"):
            results.append(res)
    return results


###########################################################
# 5) Main Script
###########################################################
if __name__ == "__main__":
    # ===================================================
    # 0) Paths & Basic Setup
    # ===================================================
    path_dataset_root = Path(r"../../../data/demo_footage")
    if not path_dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {path_dataset_root}")

    path_output = Path(r"../../../data/clipSelectionRuns")
    path_output.mkdir(exist_ok=True, parents=True)
    datetime_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    path_output = path_output / datetime_name
    path_output.mkdir(exist_ok=True, parents=True)

    plot_dir = path_output / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    clip_dir = path_output / "clips"
    clip_dir.mkdir(exist_ok=True, parents=True)

    random_clip_dir = clip_dir / "random"
    random_clip_dir.mkdir(exist_ok=True)
    cluster_clip_dir = clip_dir / "clusters"
    cluster_clip_dir.mkdir(exist_ok=True)

    n_workers = 4  # used for metadata scanning, saving

    # ===================================================
    # 1) Collect metadata
    # ===================================================
    paths_dataset_subdirs = list(path_dataset_root.iterdir())
    dict_subdir_to_video_metas = {}
    for path_dataset_subdir in paths_dataset_subdirs:
        paths_videos = (
            list(path_dataset_subdir.glob("**/*.mp4"))
            + list(path_dataset_subdir.glob("**/*.avi"))
            + list(path_dataset_subdir.glob("**/*.dv"))
        )
        video_metas_dict = {}
        with mp.Pool(n_workers) as pool:
            for path_video, fps, frame_count, size in tqdm(
                pool.imap_unordered(open_video_get_fps_frame_count_and_size, paths_videos),
                total=len(paths_videos),
                desc=f"Scanning {path_dataset_subdir.name}",
            ):
                video_metas_dict[path_video] = {
                    "fps": fps,
                    "frame_count": frame_count,
                    "size": size,
                }
        dict_subdir_to_video_metas[path_dataset_subdir] = video_metas_dict

    # (Optional) visualize or log stats
    # visualize_video_frame_lengths(plot_dir, dict_subdir_to_video_metas)
    # visualize_video_time_length_in_mins(plot_dir, dict_subdir_to_video_metas)
    # visualize_video_size_distribution(plot_dir, dict_subdir_to_video_metas)

    # ===================================================
    # 2) Decide how much time for random vs. cluster
    # ===================================================
    target_time_subsample_hours = 1.0
    total_minutes = target_time_subsample_hours * 60.0
    ratio_random = 0.5
    target_time_random = total_minutes * ratio_random
    target_time_cluster = total_minutes * (1 - ratio_random)

    # ===================================================
    # 3) Generate candidate clips
    # ===================================================
    frames_per_clip = 10
    overlap = 0.5

    candidate_clips = generate_candidate_clips(
        dict_subdir_to_video_metas,
        frames_per_clip=frames_per_clip,
        overlap=overlap,
        n_workers=n_workers,
    )
    del dict_subdir_to_video_metas
    gc.collect()

    random_pool, cluster_pool = split_random_vs_cluster(candidate_clips, ratio_random=ratio_random)
    del candidate_clips
    gc.collect()

    # ===================================================
    # 4) Sample random clips by time
    # ===================================================
    selected_random_clips = sample_random_clips_by_time(random_pool, target_time_random)
    del random_pool
    gc.collect()

    random_ids = set(map(id, selected_random_clips))
    cluster_pool = [c for c in cluster_pool if id(c) not in random_ids]
    print(f"Selected {len(selected_random_clips)} random clips for ~{target_time_random} min.")
    print(f"Remaining cluster pool: {len(cluster_pool)} clips.")

    # ===================================================
    # 5) Embed cluster pool with rolling loaders
    # ===================================================
    # We'll do e.g. 64 clips/batch. Increase if you have the GPU memory & CPU RAM.
    # We'll also set multiple loader threads, e.g. 2 or 4, to see if it helps saturate disk.
    embedded_cluster = embed_all_clips_rolling_loader(
        cluster_pool,
        clips_per_batch=64,     # try bigger e.g. 128, 256, ...
        n_loader_threads=2,     # you can also try 4, 8 if you have many CPU cores
    )
    del cluster_pool
    gc.collect()

    # Filter out None
    valid_cluster_clips = []
    cluster_embeddings = []
    for clip_info, emb in embedded_cluster:
        if emb is not None:
            valid_cluster_clips.append(clip_info)
            cluster_embeddings.append(emb)
    del embedded_cluster
    gc.collect()

    if not cluster_embeddings:
        print("No valid cluster clips found after embedding. Exiting.")
        exit(0)

    cluster_embeddings = np.vstack(cluster_embeddings)
    print("cluster_embeddings shape:", cluster_embeddings.shape)

    # ===================================================
    # 6) PCA -> UMAP
    # ===================================================
    pca_dims = 64
    ipca, pca_embeddings = do_incremental_pca(cluster_embeddings, n_components=pca_dims, batch_size=4096)
    print("pca_embeddings shape:", pca_embeddings.shape)
    del cluster_embeddings
    gc.collect()

    umap_dims = pca_dims // 4
    umap_reducer, umap_vectors = do_umap(pca_embeddings, n_components=umap_dims)
    print("umap_vectors shape:", umap_vectors.shape)
    del pca_embeddings
    gc.collect()

    # ===================================================
    # 7) Spectral Clustering
    # ===================================================
    num_clusters = 200
    cluster_labels = cluster_with_spectral(
        umap_vectors,
        n_clusters=num_clusters,
        n_neighbors=15,
        random_seed=42,
    )
    print("Spectral clustering done.", np.bincount(cluster_labels))
    del umap_vectors
    gc.collect()

    # ===================================================
    # 8) Time-based sampling from clusters
    # ===================================================
    selected_cluster_results = sample_clips_across_clusters_by_time(
        valid_cluster_clips,
        cluster_labels,
        target_minutes=target_time_cluster,
    )
    print(f"Sampled {len(selected_cluster_results)} cluster-based clips for ~{target_time_cluster} min.")
    del valid_cluster_clips
    del cluster_labels
    gc.collect()

    # ===================================================
    # 9) Save Random + Cluster Clips
    # ===================================================
    # A) Random
    random_tasks = []
    for i, clip_info in enumerate(selected_random_clips):
        random_tasks.append((clip_info, None, random_clip_dir, i))
    del selected_random_clips
    gc.collect()

    random_save_results = save_selected_clips_multiprocessing(random_tasks, n_workers=n_workers)
    num_success_random = sum(1 for (_, ok) in random_save_results if ok)
    print(f"Saved {num_success_random}/{len(random_save_results)} random clips successfully.")
    del random_tasks
    gc.collect()

    # B) Cluster-based
    cluster_tasks = []
    for i, (clip_info, c_label) in enumerate(selected_cluster_results):
        cluster_tasks.append((clip_info, c_label, cluster_clip_dir, i))
    del selected_cluster_results
    gc.collect()

    cluster_save_results = save_selected_clips_multiprocessing(cluster_tasks, n_workers=n_workers)
    num_success_cluster = sum(1 for (_, ok) in cluster_save_results if ok)
    print(f"Saved {num_success_cluster}/{len(cluster_save_results)} cluster-based clips successfully.")
    del cluster_tasks
    del cluster_save_results
    del random_save_results
    gc.collect()

    print("Done.")
