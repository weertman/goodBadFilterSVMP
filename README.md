# Video Annotation Tool

A Python application for annotating video clips with predefined classification labels. This tool is designed to streamline the video annotation process with keyboard shortcuts, randomized clip ordering, and session persistence.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Starting the Application](#starting-the-application)
  - [Setting up a Session](#setting-up-a-session)
  - [Annotating Videos](#annotating-videos)
  - [Navigation](#navigation)
  - [Playback Controls](#playback-controls)
  - [Keyboard Shortcuts](#keyboard-shortcuts)
- [Output](#output)
- [Troubleshooting](#troubleshooting)

## Overview

This tool allows users to efficiently annotate video clips by assigning predefined classes. It's particularly designed for marine biology footage classification but can be adapted for other purposes. The application tracks progress, saves annotations to CSV files, and provides various navigation and playback controls to optimize the annotation workflow.

## Features

- User-friendly Qt-based interface with video player
- Predefined classification system with keyboard shortcuts
- CSV output for annotations with timestamps and time spent
- Persistent session state (resume annotation sessions)
- Random ordering of unannotated clips
- Playback speed controls (1/2 and 1/4 speed)
- Progress tracking with visual indicators
- Direct navigation to specific clips
- Ability to jump to the last annotated clip

## Requirements

- Python 3.6+
- PySide6 (Qt for Python)
- Video codec support for MP4 files

## Installation

1. Ensure you have Python 3.6 or later installed:
   ```
   python --version
   ```

2. Install PySide6 using pip:
   ```
   pip install PySide6
   ```

3. You may need to install additional codecs depending on your operating system:
   - **Windows**: Install [K-Lite Codec Pack](https://www.codecguide.com/download_kl.htm)
   - **macOS**: Install [VLC](https://www.videolan.org/vlc/index.html)
   - **Linux**: Install appropriate gstreamer plugins:
     ```
     sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
     ```

4. Download the script (e.g., `video_annotation_tool.py`) or clone the repository.

## Directory Structure

The application expects a specific directory structure:

```
base_directory/              # Main directory selected in the app
├── clips/                  # Required subdirectory containing all video clips
│   ├── clip1.mp4           # Video clips to be annotated
│   ├── clip2.mp4
│   ├── subdirectory1/      # Optional subdirectories
│   │   ├── clip3.mp4
│   │   └── clip4.mp4
│   └── ...
└── annotations_username.csv # Generated annotation file (created by the app)
```

**Important Notes:**
- The "clips" directory is required and must contain .mp4 files
- The application can handle clips in subdirectories within the "clips" folder
- The base directory name is used as the date identifier in the application

## Usage

### Starting the Application

Run the script using Python:

```
python video_annotation_tool.py
```

### Setting up a Session

1. Enter your username in the "User Name" field
   - This name will be used to create and identify your annotation CSV file

2. Select the base directory containing the "clips" folder using the "Browse..." button
   - The application will look for a "clips" subdirectory containing .mp4 files
   - Make sure your directory structure follows the requirements outlined above

3. Click "Begin Annotation" to start

### Annotating Videos

The application presents videos for annotation with the following predefined classes:

| Class | Hotkey |
|-------|--------|
| Good | q |
| Above Ground/Veg Bad | w |
| Green Water Bad | e |
| Onshore Bad | r |
| Sunflower Star | a |
| Sea Star | s |
| Bivalves | d |
| Sea Urchin | f |
| Sand Dollar | g |

To annotate a clip:

1. Watch the video (plays automatically at 1/2 speed)
2. After the video completes playback once, classification buttons become enabled
3. Select a class either by:
   - Clicking the corresponding button
   - Pressing the corresponding keyboard shortcut
4. Once a class is selected, the "Next" button becomes enabled
5. Click "Next" or press Tab/Right Arrow to move to the next clip

### Navigation

- **Next Clip**: Click "Next" button, press Tab, or press Right Arrow
- **Previous Clip**: Click "Previous" button or press Left Arrow
- **Go to Specific Clip**: Enter a clip number in the "Go to clip" field and click "Go" or press Enter
- **Jump to Last Annotated Clip**: Click the "Last Annotated" button

### Playback Controls

- **Toggle Speed**: Switch between 1/2 speed (default) and 1/4 speed by clicking the speed button or pressing 'p'
- **Pause/Resume**: Click the "Pause" button or press Spacebar
- **Scrubbing**: When paused, a slider appears allowing you to navigate within the video

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Next clip | Tab or Right Arrow |
| Previous clip | Left Arrow |
| Toggle speed (1/2 ↔ 1/4) | p |
| Pause/Play | Spacebar |
| Close application | Escape |
| Class: Good | q |
| Class: Above Ground/Veg Bad | w |
| Class: Green Water Bad | e |
| Class: Onshore Bad | r |
| Class: Sunflower Star | a |
| Class: Sea Star | s |
| Class: Bivalves | d |
| Class: Sea Urchin | f |
| Class: Sand Dollar | g |

## Output

The application generates a CSV file named `annotations_username.csv` in the selected base directory with the following columns:

- **clip_path**: Relative path to the video clip from the base directory
- **user**: Username of the annotator
- **date**: Timestamp when the annotation was made
- **time_spent_s**: Time spent (in seconds) on the annotation
- **class**: Selected classification

Example CSV content:
```
clip_path,user,date,time_spent_s,class
clips/video1.mp4,jsmith,2025-04-01 14:23:45,12.345,Good
clips/subdir/video2.mp4,jsmith,2025-04-01 14:24:12,8.765,Sea Star
```

## Troubleshooting

- **Video won't play**: Ensure you have the necessary codecs installed for MP4 playback
- **"No 'clips' directory found" error**: Make sure your selected directory contains a subdirectory named "clips"
- **"Cannot find clip file" error**: The file referenced in the CSV may have been moved or deleted
- **UI elements not responding to Tab key**: The application captures Tab for navigation between clips
- **Annotations not saving**: Check write permissions in the selected directory
