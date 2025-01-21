import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_video_frame_lengths (plot_dir, dict_subdir_to_video_metas):
    ## visualize distribution of video lengths in frames
    video_lengths = []
    for video_metas_dict in dict_subdir_to_video_metas.values():
        for video_meta in video_metas_dict.values():
            video_lengths.append(video_meta['frame_count'])

    sum_video_lengths = np.sum(video_lengths)
    mean_video_lengths = np.mean(video_lengths)
    min_video_length = np.min(video_lengths)
    max_video_length = np.max(video_lengths)
    std_video_length = np.std(video_lengths)

    print(f'Number of videos: {len(video_lengths)}')
    print(f'Total number of frames: {sum_video_lengths}')
    print(f'Min: {min_video_length}, Max: {max_video_length}, Mean: {mean_video_lengths}, Std: {std_video_length}')

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(video_lengths, bins=100)
    ax.set_xlabel('Video length in frames')
    ax.set_ylabel('Frequency')
    title = f'Min: {min_video_length}, Max: {max_video_length}, Mean: {np.round(mean_video_lengths,4)}, Std: {np.round(std_video_length,4)}, Total: {sum_video_lengths}'
    ax.set_title(title)
    fig.savefig(plot_dir / 'video_lengths_frames.png', dpi=300)
    plt.close(fig)

def visualize_video_time_length_in_mins (plot_dir, dict_subdir_to_video_metas):
    ## visualize distribution of video time lengths in frames
    video_lengths = []
    for video_metas_dict in dict_subdir_to_video_metas.values():
        for video_meta in video_metas_dict.values():
            video_length = video_meta['frame_count'] / video_meta['fps'] / 60
            video_lengths.append(video_length)

    sum_video_lengths = np.sum(video_lengths)
    mean_video_lengths = np.mean(video_lengths)
    min_video_length = np.min(video_lengths)
    max_video_length = np.max(video_lengths)
    std_video_length = np.std(video_lengths)

    print(f'Number of videos: {len(video_lengths)}')
    print(f'Total number of minutes: {sum_video_lengths}')
    print(f'Min: {min_video_length}, Max: {max_video_length}, Mean: {mean_video_lengths}, Std: {std_video_length}')

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(video_lengths, bins=100)
    ax.set_xlabel('Video length in minutes')
    ax.set_ylabel('Frequency')
    title = f'Min: {min_video_length}, Max: {max_video_length}, Mean: {np.round(mean_video_lengths, 4)}, Std: {np.round(std_video_length, 4)}, Total: {sum_video_lengths}'
    ax.set_title(title)
    fig.savefig(plot_dir / 'video_lengths_minutes.png', dpi=300)
    plt.close(fig)

def visualize_video_size_distribution (plot_dir, dict_subdir_to_video_metas):
    # visualize distribution of video resolutions
    video_widths, video_heights = [], []
    for video_metas_dict in dict_subdir_to_video_metas.values():
        for video_meta in video_metas_dict.values():
            video_widths.append(video_meta['size'][0])
            video_heights.append(video_meta['size'][1])

    resolutions = [w*h for w, h in zip(video_widths, video_heights)]

    mean_video_width = np.mean(video_widths)
    mean_video_height = np.mean(video_heights)
    min_video_width = np.min(video_widths)
    min_video_height = np.min(video_heights)
    max_video_width = np.max(video_widths)
    max_video_height = np.max(video_heights)
    std_video_width = np.std(video_widths)
    std_video_height = np.std(video_heights)

    mean_resolution = np.mean(resolutions)
    min_resolution = np.min(resolutions)
    max_resolution = np.max(resolutions)
    std_resolution = np.std(resolutions)

    print(f'Mean video width: {mean_video_width}, Mean video height: {mean_video_height}')
    print(f'Min video width: {min_video_width}, Min video height: {min_video_height}')
    print(f'Max video width: {max_video_width}, Max video height: {max_video_height}')
    print(f'Std video width: {std_video_width}, Std video height: {std_video_height}')

    print(f'Mean resolution: {mean_resolution}')
    print(f'Min resolution: {min_resolution}')
    print(f'Max resolution: {max_resolution}')
    print(f'Std resolution: {std_resolution}')

    unique_width_heights = np.unique(np.array([video_widths, video_heights]).T, axis=0)
    for width, height in unique_width_heights:
        print(f'Num videos @ wxh of {width}x{height} is {np.sum(np.logical_and(np.array(video_widths) == width, np.array(video_heights) == height))}')

    fig, axs = plt.subplots(2,2, figsize=(15, 15))

    axs[0,0].hist(video_widths, bins=100)
    axs[0,0].set_xlabel('Video width')

    axs[0,1].hist(video_heights, bins=100)
    axs[0,1].set_xlabel('Video height')

    axs[1,0].scatter(video_widths, video_heights, alpha=0.5)
    axs[1,0].set_xlabel('Video width')
    axs[1,0].set_ylabel('Video height')

    axs[1,1].hist(resolutions, bins=100)
    axs[1,1].set_xlabel('Video resolution')

    fig.savefig(plot_dir / 'video_resolutions.png', dpi=300)
    plt.close(fig)