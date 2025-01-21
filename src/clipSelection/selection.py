import os
from pathlib import Path
from tqdm import tqdm
import cv2
from datetime import datetime
import multiprocessing as mp
import numpy as np
import torch
import random
from PIL import Image
import clip
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
import umap
import subprocess
import pickle
import gc

# Your custom plotting functions
from visualizations import (
    visualize_video_frame_lengths,
    visualize_video_time_length_in_mins,
    visualize_video_size_distribution,
)

########################################################################
# Global variables for each worker
########################################################################
DEVICE = None
MODEL = None
PREPROCESS = None
RUN_CONFIG = None

########################################################################
# 0) open_video_get_fps_frame_count_and_size
########################################################################
def open_video_get_fps_frame_count_and_size(path_video):
    cap = cv2.VideoCapture(str(path_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    cap.release()
    return path_video, fps, frame_count, size

########################################################################
# 1) Generate Candidate Clips
########################################################################
def generate_candidate_clips_for_video(args, frames_per_clip=10, overlap=0.5):
    path_video, meta = args
    fps = meta["fps"]
    frame_count = meta["frame_count"]
    width, height = meta["size"]

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
            "width": width,
            "height": height,
            "duration_seconds": clip_length / fps,
        }
        candidate_clips.append(clip_info)
        start += step

    return candidate_clips

def generate_candidate_clips(dict_subdir_to_video_metas, frames_per_clip=10, overlap=0.5, n_workers=4):
    from functools import partial

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

########################################################################
# 2) Split Random vs. Cluster
########################################################################
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

########################################################################
# 3) Time-based sampling across clusters
########################################################################
def sample_clips_across_clusters_by_time(valid_clips, cluster_labels, target_minutes=30.0):
    from collections import defaultdict

    cluster_to_indices = defaultdict(list)
    for i, c in enumerate(cluster_labels):
        cluster_to_indices[c].append(i)

    # Shuffle index lists
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

########################################################################
# 4) Single-pass embedding logic (with sidecar)
########################################################################
def collect_needed_frames_for_video(clips_for_this_video):
    needed = set()
    for clip in clips_for_this_video:
        s = clip["start_frame"]
        e = clip["end_frame"]
        if e > s:
            m = s + (e - s)//2
            needed.add(s)
            needed.add(m)
            needed.add(e - 1)
    return needed

def read_frames_ffmpeg_needed(video_path, max_needed_frame, width, height,
                              needed_frames_set, model, preprocess, device,
                              show_progress=False):
    from tqdm import tqdm as tqdm_local

    frame_embeddings = {}
    if max_needed_frame < 0:
        return frame_embeddings

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-err_detect", "explode",
        "-ss", "0",
        "-i", str(video_path),
        "-frames:v", str(max_needed_frame + 1),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    bytes_per_frame = width * height * 3

    frame_idx = 0
    pbar = None
    if show_progress:
        pbar = tqdm_local(total=max_needed_frame + 1, desc=f"Decoding {video_path.name}", leave=False)

    while True:
        raw_frame = process.stdout.read(bytes_per_frame)
        if not raw_frame or len(raw_frame) < bytes_per_frame:
            break

        if frame_idx in needed_frames_set:
            frame_rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            pil_img = Image.fromarray(frame_rgb)
            with torch.no_grad():
                img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                emb = model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            frame_embeddings[frame_idx] = emb.cpu().numpy()

        frame_idx += 1
        if pbar is not None:
            pbar.update(1)

        if frame_idx > max_needed_frame:
            break

    if pbar is not None:
        pbar.close()

    process.stdout.close()
    process.stderr.close()
    process.wait()
    return frame_embeddings

def compute_clip_embeddings_for_video(clips_for_this_video, frame_embeds_dict):
    results = []
    for clip in clips_for_this_video:
        s = clip["start_frame"]
        e = clip["end_frame"]
        if e <= s:
            results.append((clip, None))
            continue

        m = s + (e - s)//2
        frames_to_check = [s, m, e - 1]
        valid = [frame_embeds_dict[idx] for idx in frames_to_check if idx in frame_embeds_dict]
        if not valid:
            results.append((clip, None))
        else:
            arr = np.vstack(valid)
            clip_emb = arr.mean(axis=0)
            results.append((clip, clip_emb))
    return results

########################################################################
# Sidecar saving/loading
########################################################################
def get_sidecar_path(video_path: Path) -> Path:
    return video_path.with_suffix('.emb.npz')

def sidecar_metadata_matches_run_config(metadata: dict, run_config: dict) -> bool:
    fields_to_check = ["frames_per_clip", "overlap", "model_name"]
    for f in fields_to_check:
        if metadata.get(f) != run_config.get(f):
            return False
    return True

def load_sidecar_if_compatible(video_path: Path, run_config: dict):
    sidecar_path = get_sidecar_path(video_path)
    if not sidecar_path.exists():
        return None
    try:
        with np.load(sidecar_path, allow_pickle=True) as data:
            metadata = pickle.loads(data['metadata'].item())
            if not sidecar_metadata_matches_run_config(metadata, run_config):
                return None
            clip_embeds = pickle.loads(data['clip_embeddings'].item())
            return clip_embeds
    except Exception as e:
        print(f"[WARN] Could not load sidecar for {video_path.name}: {e}")
        return None

def save_sidecar(video_path: Path, run_config: dict, clip_embeds: dict):
    sidecar_path = get_sidecar_path(video_path)
    to_save = {
        'metadata': np.array([pickle.dumps(run_config)], dtype=object),
        'clip_embeddings': np.array([pickle.dumps(clip_embeds)], dtype=object),
    }
    np.savez_compressed(sidecar_path, **to_save)

def embed_clips_for_one_video_with_sidecar(video_path, clips_for_video, meta, show_progress=False):
    global DEVICE, MODEL, PREPROCESS, RUN_CONFIG

    existing_embeds = load_sidecar_if_compatible(video_path, RUN_CONFIG)
    if existing_embeds is None:
        existing_embeds = {}

    def clip_key(c):
        return (c["start_frame"], c["end_frame"])

    needed_clips = [c for c in clips_for_video if clip_key(c) not in existing_embeds]

    if not needed_clips:
        # all done previously
        return [(c, existing_embeds[clip_key(c)]) for c in clips_for_video]

    width, height = meta["size"]
    needed_frames = collect_needed_frames_for_video(needed_clips)
    max_needed_frame = max(needed_frames) if needed_frames else -1

    print(f"[DEBUG] About to decode: {video_path}")
    print(f"[DEBUG] Needed clips: {len(needed_clips)}, Max needed frame: {max_needed_frame}")

    frame_embeds_dict = read_frames_ffmpeg_needed(
        video_path=video_path,
        max_needed_frame=max_needed_frame,
        width=width,
        height=height,
        needed_frames_set=needed_frames,
        model=MODEL,
        preprocess=PREPROCESS,
        device=DEVICE,
        show_progress=show_progress
    )

    new_results = compute_clip_embeddings_for_video(needed_clips, frame_embeds_dict)
    for clip_info, emb in new_results:
        existing_embeds[clip_key(clip_info)] = emb

    save_sidecar(video_path, RUN_CONFIG, existing_embeds)

    return [(c, existing_embeds[clip_key(c)]) for c in clips_for_video]

########################################################################
# 5) Multiprocessing driver
########################################################################
def init_worker(run_config):
    global DEVICE, MODEL, PREPROCESS, RUN_CONFIG

    RUN_CONFIG = run_config

    device_str = run_config.get("device_str", "cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device(device_str)

    model_name = run_config.get("model_name", "ViT-L/14")
    MODEL, PREPROCESS = clip.load(model_name, device=DEVICE)
    MODEL.eval()

def embed_clips_for_one_video(args):
    (video_path, clips_for_video, meta, show_progress) = args
    return embed_clips_for_one_video_with_sidecar(video_path, clips_for_video, meta, show_progress)

def group_clips_by_video(candidate_clips):
    from collections import defaultdict
    video_to_clips = defaultdict(list)
    for clip in candidate_clips:
        video_to_clips[clip["video_path"]].append(clip)
    return video_to_clips

def embed_cluster_pool_by_video(cluster_pool, run_config, n_workers=4, show_progress=False):
    """
    - Group cluster clips by video
    - Create a single Pool, embed each video, gather results
    - No timeouts: if a video is large/slow, we wait until it finishes
    - If you stop the script mid-run, sidecar files for completed videos
      will remain, so a restart will skip those already done.
    """
    video_to_clips = group_clips_by_video(cluster_pool)
    tasks = []
    for video_path, clips in video_to_clips.items():
        if not clips:
            continue
        meta = {
            "fps": clips[0]["fps"],
            "size": (clips[0]["width"], clips[0]["height"]),
        }
        tasks.append((video_path, clips, meta, show_progress))

    results_all = []
    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(run_config,)) as pool:
        for single_video_result in tqdm(
            pool.imap_unordered(embed_clips_for_one_video, tasks),
            total=len(tasks),
            desc="Embedding By Video"
        ):
            results_all.extend(single_video_result)

    return results_all

########################################################################
# 6) PCA + UMAP + MBK-Means
########################################################################
def do_incremental_pca(embeddings, n_components=64, batch_size=4096):
    from sklearn.decomposition import IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    n_samples = embeddings.shape[0]

    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        ipca.partial_fit(embeddings[start_idx:end_idx])

    pca_embeddings_list = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        pca_embeddings_list.append(ipca.transform(embeddings[start_idx:end_idx]))

    pca_embeddings = np.vstack(pca_embeddings_list)
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

def cluster_with_minibatch_kmeans(embeddings, n_clusters=200, random_seed=42, batch_size=4096):
    mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_seed)
    mbk.fit(embeddings)
    return mbk.labels_

########################################################################
# 7) Save Clips
########################################################################
def read_frames_ffmpeg(video_path, start_frame, end_frame, fps, width, height):
    frame_count = end_frame - start_frame
    if frame_count <= 0:
        return

    start_time_seconds = start_frame / float(fps)

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-err_detect", "explode",
        "-ss", str(start_time_seconds),
        "-i", str(video_path),
        "-frames:v", str(frame_count),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    bytes_per_frame = width * height * 3

    for _ in range(frame_count):
        raw_frame = process.stdout.read(bytes_per_frame)
        if not raw_frame or len(raw_frame) < bytes_per_frame:
            break
        frame_rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
        yield frame_rgb

    process.stdout.close()
    process.stderr.close()
    process.wait()

def save_one_clip(args):
    clip_info, cluster_label, base_output_dir, idx = args

    video_path = clip_info["video_path"]
    start_frame = clip_info["start_frame"]
    end_frame = clip_info["end_frame"]
    fps = clip_info["fps"]
    width = clip_info["width"]
    height = clip_info["height"]

    subdir_name = f"clip_{idx:05d}"
    if cluster_label is not None:
        subdir_name += f"_cluster_{cluster_label}"
    output_subdir = base_output_dir / subdir_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem
    out_video_path = output_subdir / f"{video_name}_clip_{start_frame}-{end_frame}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    written_count = 0
    for frame_rgb in read_frames_ffmpeg(video_path, start_frame, end_frame, fps, width, height):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out_writer.write(frame_bgr)
        written_count += 1

    out_writer.release()
    success = (written_count > 0)
    return (idx, success)

def save_selected_clips_multiprocessing(tasks, n_workers=4):
    results = []
    with mp.Pool(n_workers) as pool:
        for res in tqdm(pool.imap_unordered(save_one_clip, tasks), total=len(tasks), desc="Saving Clips"):
            results.append(res)
    return results

########################################################################
# 8) Main Script
########################################################################
if __name__ == "__main__":
    path_dataset_root = Path(r'D:\StarSurvey\data\SVMP_footage')
    if not path_dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {path_dataset_root}")

    path_output = Path(r"../../data/SVMP_footage")
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

    n_workers = 2

    ###################################################################
    # 1) Collect metadata
    ###################################################################
    paths_dataset_subdirs = list(path_dataset_root.iterdir())
    dict_subdir_to_video_metas = {}

    for path_dataset_subdir in paths_dataset_subdirs:
        paths_videos = (
            list(path_dataset_subdir.glob("**/*.mp4"))
            + list(path_dataset_subdir.glob("**/*.avi"))
            + list(path_dataset_subdir.glob("**/*.dv"))
        )
        if not paths_videos:
            continue

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

    # Optional: plotting
    visualize_video_frame_lengths(plot_dir, dict_subdir_to_video_metas)
    visualize_video_time_length_in_mins(plot_dir, dict_subdir_to_video_metas)
    visualize_video_size_distribution(plot_dir, dict_subdir_to_video_metas)

    ###################################################################
    # 2) Decide how much time we want for random vs. cluster
    ###################################################################
    target_time_subsample_hours = 4.2
    total_minutes = target_time_subsample_hours * 60.0
    ratio_random = 0
    target_time_random = total_minutes * ratio_random
    target_time_cluster = total_minutes * (1 - ratio_random)

    ###################################################################
    # 3) Generate all candidate clips
    ###################################################################
    frames_per_clip = 16
    overlap = 0.1

    candidate_clips = generate_candidate_clips(
        dict_subdir_to_video_metas,
        frames_per_clip=frames_per_clip,
        overlap=overlap,
        n_workers=n_workers,
    )

    del dict_subdir_to_video_metas
    gc.collect()

    ###################################################################
    # Split into random vs. cluster pools
    ###################################################################
    random_pool, cluster_pool = split_random_vs_cluster(candidate_clips, ratio_random=ratio_random)
    del candidate_clips
    gc.collect()

    ###################################################################
    # 4) Sample random clips by time
    ###################################################################
    selected_random_clips = sample_random_clips_by_time(random_pool, target_time_random)
    del random_pool
    gc.collect()

    # Remove selected random from cluster pool
    random_ids = set(map(id, selected_random_clips))
    cluster_pool = [c for c in cluster_pool if id(c) not in random_ids]

    print(f"Selected {len(selected_random_clips)} random clips covering ~{target_time_random} min.")
    print(f"Remaining cluster pool: {len(cluster_pool)} clips.")

    ###################################################################
    # 5) Embedding with sidecar
    ###################################################################
    run_config = {
        "frames_per_clip": frames_per_clip,
        "overlap": overlap,
        "model_name": "ViT-L/14",
        "device_str": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    show_per_video_progress = True
    embedded_cluster = embed_cluster_pool_by_video(
        cluster_pool,
        run_config=run_config,
        n_workers=n_workers,
        show_progress=show_per_video_progress
    )
    del cluster_pool
    gc.collect()

    # Filter out None embeddings
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

    ###################################################################
    # 6) PCA -> UMAP
    ###################################################################
    pca_dims = 64
    ipca, pca_embeddings = do_incremental_pca(cluster_embeddings, n_components=pca_dims, batch_size=4096)
    print("pca_embeddings shape:", pca_embeddings.shape)

    del cluster_embeddings
    gc.collect()

    umap_dims = pca_dims // 4  # e.g. 16
    reducer, umap_vectors = do_umap(pca_embeddings, n_components=umap_dims)
    print("umap_vectors shape:", umap_vectors.shape)

    del pca_embeddings
    gc.collect()

    ###################################################################
    # 7) Mini-Batch K-Means
    ###################################################################
    num_clusters = 420
    cluster_labels = cluster_with_minibatch_kmeans(
        umap_vectors,
        n_clusters=num_clusters,
        random_seed=42,
        batch_size=4096
    )
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("Mini-Batch K-Means done.", dict(zip(unique, counts)))

    del umap_vectors
    gc.collect()

    ###################################################################
    # 8) Time-based sampling from clusters
    ###################################################################
    selected_cluster_results = sample_clips_across_clusters_by_time(
        valid_cluster_clips,
        cluster_labels,
        target_minutes=target_time_cluster,
    )
    print(f"Sampled {len(selected_cluster_results)} cluster-based clips covering ~{target_time_cluster} min.")

    del valid_cluster_clips, cluster_labels
    gc.collect()

    ###################################################################
    # 9) Save Random + Cluster Clips
    ###################################################################
    # A) Random
    random_tasks = []
    for i, clip_info in enumerate(selected_random_clips):
        random_tasks.append((clip_info, None, Path(random_clip_dir), i))
    del selected_random_clips
    gc.collect()

    random_save_results = save_selected_clips_multiprocessing(random_tasks, n_workers=n_workers)
    num_success_random = sum(1 for (_, ok) in random_save_results if ok)
    print(f"Saved {num_success_random}/{len(random_save_results)} random clips successfully.")
    del random_tasks, random_save_results
    gc.collect()

    # B) Cluster-based
    cluster_tasks = []
    for i, (clip_info, c_label) in enumerate(selected_cluster_results):
        cluster_tasks.append((clip_info, c_label, Path(cluster_clip_dir), i))
    del selected_cluster_results
    gc.collect()

    cluster_save_results = save_selected_clips_multiprocessing(cluster_tasks, n_workers=n_workers)
    num_success_cluster = sum(1 for (_, ok) in cluster_save_results if ok)
    print(f"Saved {num_success_cluster}/{len(cluster_save_results)} cluster-based clips successfully.")
    del cluster_tasks, cluster_save_results
    gc.collect()

    print("Done.")
