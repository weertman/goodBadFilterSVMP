# Good/Bad Filter for SVMP Dataset

This repository provides a simple **video annotation tool** intended to classify short video clips as "Good" or "Bad" (along with other categories) for the SVMP dataset.

---

## Installation

### 1. Clone or Download This Repository

```bash
git clone https://github.com/YourUserName/goodBadFilterSVMP.git
cd goodBadFilterSVMP
```
### 2. Create and Activate a Conda Environment
```bash
conda create -n goodBadFilterSVMP python=3.9
conda activate goodBadFilterSVMP
```

### 3. Install Dependancies
```bash
pip install pyside6
```

### 4. Download clips from google drive
I've pushed 4.2 hours of clips to the drive for you to annotate.
Link: https://drive.google.com/drive/folders/10wNXE3BPpo8VcOLY5wu4TpcM48z6dBSr?usp=sharing
Or from dropbox if drive does not work..
Link: https://www.dropbox.com/scl/fi/jtooi0c3bpny9os2klszm/20250119-175129.zip?rlkey=55j8p03uww3kvfmjeertsnqdd&st=d0e6c45a&dl=0

## 5. Launch the tool
```bash
python src/annotator/interface.py
```

## Choose Username & Video Directory

When the GUI opens:

1. **Enter a user name** in the **"User Name"** field.  
2. **Browse or type in** the directory containing your `.mp4` files.

---

## Begin Annotation

Click the **"Begin Annotation"** button.

- The script gathers all `.mp4` files in the directory (and its subfolders).
- A CSV file named `annotations_<username>.csv` is created or updated in that directory.
- The tool displays each video, in a random order, for classification.

---

## Annotate

1. **Wait** until at least one loop of the video completes *(the "Next" button is disabled until you watch it once)*.
2. **Click** one of the class buttons or press the associated hotkey (e.g., `q`, `w`, `e`, etc.).  
   - The selected button will highlight to confirm your choice.
3. **Press Tab** (or click **"Next"**) to move on to the next clip.

---

## Playback Controls

- **Pause**: Press <kbd>Space</kbd> or click the **"Pause (space)"** button.  
- **Speed Toggle**: Press <kbd>p</kbd> or click the **"1/4 Speed (p)"** button to toggle between 1/2 and 1/4 speed.  
- **Scrubbing**: While paused, a slider appears that you can drag to seek in the video.

Repeat until no more clips remain. A **"Done"** message appears when all clips have been annotated.

---

## CSV Output

Every time you click **Next**, a row is appended to the CSV file located in your chosen video directory (named `annotations_<username>.csv`). Each row contains:

- **`clip_path`**: The relative path to the video file.  
- **`user`**: The username you entered at the start.  
- **`date`**: The date/time at which the annotation was saved.  
- **`time_spent_s`**: How many seconds you spent before hitting **Next**.  
- **`class`**: The selected class/category (e.g., `"Good"` or `"Above Ground/Veg Bad"` etc.).  
