import sys
import os
import random
import csv
import time
from datetime import datetime

from PySide6.QtCore import Qt, QUrl, Slot, QTimer
from PySide6.QtGui import QShortcut, QKeySequence, QIntValidator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLineEdit, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QSlider, QProgressBar,
    QStyleFactory
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

##############################################################################
# 1) Define your classes and hotkeys here
##############################################################################
# Format: [("ClassName", "Hotkey"), ...]
CLASS_DEFINITIONS = [
    ("Good", "q"),
    ("Above Ground/Veg Bad", "w"),
    ("Green Water Bad", "e"),
    ("Onshore Bad", "r"),
    ("Sunflower Star", "a"),
    ("Sea Star", "s"),
    ("Bivalves", "d"),
    ("Sea Urchin", "f"),
    ("Sand Dollar", "g"),
]


##############################################################################
# 2) Main Window
##############################################################################
class AnnotationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Annotation Tool")

        # Let this window capture tab if no child grabs focus
        self.setFocusPolicy(Qt.StrongFocus)

        # --- State variables ---
        self.user_name = ""
        self.clip_dir = ""
        self.clip_paths = []  # The random-ordered list of relative .mp4 paths
        self.current_index = 0  # Which clip we are on
        self.csv_path = ""  # Path to the CSV file that records annotations
        self.start_time = 0.0  # For measuring how long user spent on a clip
        self.has_watched_clip = False
        self.chosen_class = None
        self.annotations = {}  # Dictionary to store annotations temporarily

        # Toggle playback rate between 0.5 (1/2 speed) and 0.25 (1/4 speed)
        self.is_quarter_speed = False
        self.is_paused = False

        # Track how many clips have been annotated so far
        self.ann_count = 0

        # --- Widgets for "Setup" (username + directory) ---
        self.user_name_label = QLabel("User Name:")
        self.user_name_edit = QLineEdit()

        self.clip_dir_label = QLabel("Clip Directory:")
        self.clip_dir_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.on_browse)

        self.begin_button = QPushButton("Begin Annotation")
        self.begin_button.clicked.connect(self.on_begin)

        # Setup layout
        setup_layout = QVBoxLayout()
        setup_layout.addWidget(self.user_name_label)
        setup_layout.addWidget(self.user_name_edit)
        h_dir_layout = QHBoxLayout()
        h_dir_layout.addWidget(self.clip_dir_label)
        h_dir_layout.addWidget(self.clip_dir_edit)
        h_dir_layout.addWidget(self.browse_button)
        setup_layout.addLayout(h_dir_layout)
        setup_layout.addWidget(self.begin_button)

        self.setup_widget = QWidget()
        self.setup_widget.setLayout(setup_layout)

        # --- Annotation widgets ---
        self.video_widget = QVideoWidget()
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video_widget)

        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        # Connect signals
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)

        # Progress indicator with direct navigation
        self.progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Clip 0 of 0")

        # Add direct navigation input
        self.go_to_layout = QHBoxLayout()
        self.go_to_label = QLabel("Go to clip:")
        self.go_to_input = QLineEdit()
        self.go_to_input.setFixedWidth(50)  # Narrow input box
        self.go_to_input.setValidator(QIntValidator(1, 9999))  # Only allow integers
        self.go_to_button = QPushButton("Go")
        self.go_to_button.clicked.connect(self.on_go_to_clip)
        self.go_to_input.returnPressed.connect(self.on_go_to_clip)  # Allow pressing Enter

        # Add "Last" button to jump to the last annotated clip
        self.go_to_last_button = QPushButton("Last Annotated")
        self.go_to_last_button.clicked.connect(self.on_go_to_last_annotated)
        self.go_to_last_button.setToolTip("Jump to the last annotated clip")

        self.go_to_layout.addWidget(self.go_to_label)
        self.go_to_layout.addWidget(self.go_to_input)
        self.go_to_layout.addWidget(self.go_to_button)
        self.go_to_layout.addWidget(self.go_to_last_button)

        self.progress_layout.addLayout(self.go_to_layout)
        self.progress_layout.addWidget(self.progress_label)

        # Progress bar without animation
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        # Disable progress bar animation
        self.progress_bar.setStyle(QStyleFactory.create("Fusion"))  # Use Fusion style which has less animation
        self.progress_layout.addWidget(self.progress_bar)

        # Prepare a dict to map class_name -> button for easy highlight
        self.class_button_map = {}

        # Class buttons
        self.class_buttons = []
        for class_name, hotkey in CLASS_DEFINITIONS:
            btn = QPushButton(f"{class_name} ({hotkey})")
            btn.setEnabled(False)
            btn.setCheckable(True)  # Make the button checkable (highlights on select)
            self.class_button_map[class_name] = btn

            btn.clicked.connect(lambda checked, cn=class_name: self.on_class_chosen(cn))
            self.class_buttons.append(btn)

            # Shortcut for class hotkey
            shortcut = QShortcut(QKeySequence(hotkey), self)
            shortcut.activated.connect(lambda cn=class_name: self.on_class_chosen(cn))

        # Navigation buttons with keyboard shortcuts
        self.prev_button = QPushButton("Previous (←)")
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self.on_prev_clicked)
        prev_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        prev_shortcut.activated.connect(self.on_prev_clicked)

        # "Next" button (triggered by Tab or Right arrow)
        self.next_button = QPushButton("Next (→)")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.on_next_clicked)
        next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        next_shortcut.activated.connect(self.on_next_clicked)

        # Speed toggle (hotkey 'p')
        self.speed_toggle_button = QPushButton("1/4 Speed (p)")
        self.speed_toggle_button.clicked.connect(self.toggle_speed)
        speed_shortcut = QShortcut(QKeySequence("p"), self)
        speed_shortcut.activated.connect(self.toggle_speed)

        # Pause/resume (hotkey 'Space')
        self.pause_button = QPushButton("Pause (space)")
        self.pause_button.clicked.connect(self.toggle_pause)
        pause_shortcut = QShortcut(QKeySequence("Space"), self)
        pause_shortcut.activated.connect(self.toggle_pause)

        # Close
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_app)
        close_shortcut = QShortcut(QKeySequence("Escape"), self)
        close_shortcut.activated.connect(self.close_app)

        # Annotation layout
        annotation_layout = QVBoxLayout()
        annotation_layout.addWidget(self.video_widget)

        # Position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setSingleStep(50)
        self.position_slider.setPageStep(200)
        self.position_slider.setEnabled(False)
        self.position_slider.setVisible(False)
        self.position_slider.valueChanged.connect(self.on_slider_moved)
        annotation_layout.addWidget(self.position_slider)

        # Add progress indicator
        annotation_layout.addLayout(self.progress_layout)

        # Class buttons row
        class_buttons_layout = QHBoxLayout()
        for btn in self.class_buttons:
            class_buttons_layout.addWidget(btn)
        annotation_layout.addLayout(class_buttons_layout)

        # Bottom row: speed toggle + pause + prev + next + close
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.speed_toggle_button)
        bottom_layout.addWidget(self.pause_button)
        bottom_layout.addWidget(self.prev_button)
        bottom_layout.addWidget(self.next_button)
        bottom_layout.addWidget(self.close_button)
        annotation_layout.addLayout(bottom_layout)

        self.annotation_widget = QWidget()
        self.annotation_widget.setLayout(annotation_layout)
        self.annotation_widget.setVisible(False)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.setup_widget)
        main_layout.addWidget(self.annotation_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    ##########################################################################
    # 2.1) Capturing Tab in keyPressEvent
    ##########################################################################
    def keyPressEvent(self, event):
        """
        If user presses Tab while in annotation mode,
        we call self.on_next_clicked() instead of default focus navigation.
        """
        # Skip special handling if the go_to_input has focus (allow normal typing)
        if self.go_to_input.hasFocus():
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_Tab:
            # Trigger the Next button if it's enabled
            if self.next_button.isEnabled():
                self.on_next_clicked()
            event.accept()  # Prevent normal Tab focus navigation
        else:
            super().keyPressEvent(event)

    ##########################################################################
    # 2.2) Setup methods
    ##########################################################################
    @Slot()
    def on_browse(self):
        chosen_dir = QFileDialog.getExistingDirectory(self, "Select Clip Directory")
        if chosen_dir:
            self.clip_dir_edit.setText(chosen_dir)

    @Slot()
    def on_begin(self):
        self.user_name = self.user_name_edit.text().strip()
        self.clip_dir = self.clip_dir_edit.text().strip()

        if not self.user_name:
            QMessageBox.warning(self, "Error", "Please enter a valid user name.")
            return
        if not os.path.isdir(self.clip_dir):
            QMessageBox.warning(self, "Error", "Clip directory is invalid.")
            return

        # Extract the date folder name from the selected directory path
        self.date_folder_name = os.path.basename(os.path.normpath(self.clip_dir))
        print(f"Using date folder: {self.date_folder_name}")

        self.csv_path = os.path.join(self.clip_dir, f"annotations_{self.user_name}.csv")

        # Check if there's a clips directory under the selected folder
        clips_dir = os.path.join(self.clip_dir, "clips")
        if not os.path.isdir(clips_dir):
            QMessageBox.warning(self, "Error", "No 'clips' directory found under the selected folder.")
            return

        # Gather .mp4 files only within the "clips" directory and its subdirectories
        all_clips = []
        # Create a mapping of filenames to paths for quick lookup
        filename_to_path = {}

        for root, dirs, files in os.walk(clips_dir):
            for fname in files:
                if fname.lower().endswith(".mp4"):
                    rel_path = os.path.relpath(os.path.join(root, fname), self.clip_dir)
                    all_clips.append(rel_path)
                    # Store the mapping of filename to path
                    filename_to_path[fname] = rel_path

        if not all_clips:
            QMessageBox.information(self, "No Clips", "No .mp4 clips found in the clips directory.")
            return

        print(f"Found {len(all_clips)} eligible clips in the clips directory.")

        # Load or create CSV
        if os.path.exists(self.csv_path):
            existing_annotations = {}  # Maps filename to (path, class)
            annotated_paths = []  # List to preserve order of annotated clips

            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract the filename from the path
                    path = row["clip_path"]
                    filename = os.path.basename(path)

                    # Only include clips that have a classification
                    if row["class"]:
                        if filename in filename_to_path:
                            # Use the current path for this filename, not the one from CSV
                            current_path = filename_to_path[filename]
                            existing_annotations[filename] = (current_path, row["class"])
                            annotated_paths.append(current_path)
                            # Store the annotation for this path
                            self.annotations[current_path] = row["class"]
                        else:
                            print(f"Skipping annotation for file not in clips directory: {filename}")

            # Check which clips still need annotation by looking at filenames
            annotated_filenames = set(existing_annotations.keys())
            unannotated_clips = []

            for rel_path in all_clips:
                filename = os.path.basename(rel_path)
                if filename not in annotated_filenames:
                    unannotated_clips.append(rel_path)

            print(f"Already annotated: {len(annotated_paths)} clips")
            print(f"Remaining to annotate: {len(unannotated_clips)} clips")

            # Randomize the new clips
            random.shuffle(unannotated_clips)

            # Use the current paths for annotated clips + unannotated clips
            self.clip_paths = annotated_paths + unannotated_clips
        else:
            self.clip_paths = list(all_clips)
            random.shuffle(self.clip_paths)
            self.annotations = {}  # Initialize empty annotations dictionary

            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["clip_path", "user", "date", "time_spent_s", "class"]
                )
                writer.writeheader()

        # Count how many are annotated
        self.ann_count = len(self.annotations)

        print(f"Total clips to process: {len(self.clip_paths)}")
        print(f"Already annotated: {self.ann_count}")

        # Switch to annotation mode
        self.setup_widget.setVisible(False)
        self.annotation_widget.setVisible(True)

        # Make sure annotation widgets can't receive tab-focus (with exception for go_to_input)
        # so that our keyPressEvent will see Tab.
        for child in self.annotation_widget.findChildren(QWidget):
            if child != self.go_to_input:  # Exclude our navigation input box
                child.setFocusPolicy(Qt.NoFocus)
            else:
                child.setFocusPolicy(Qt.StrongFocus)  # Ensure our input box can receive focus

        # Give the main window focus so it receives key events
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # Find the actual first unannotated clip by searching through the clip_paths
        found_unannotated = False
        for i, path in enumerate(self.clip_paths):
            if path not in self.annotations:
                self.current_index = i
                found_unannotated = True
                print(f"Starting at first unannotated clip: index {i}")
                break

        if not found_unannotated:
            self._show_done_message()
            return

        # Set up progress bar
        self.progress_bar.setRange(0, len(self.clip_paths))
        self.progress_bar.setValue(self.ann_count)

        # Set up validator max value for go-to input
        self.go_to_input.setValidator(QIntValidator(1, len(self.clip_paths)))

        self.load_current_clip()

    def _get_annotated_clips(self):
        """
        Returns a set of clip paths that have already been annotated.
        Note: This uses the paths stored in self.annotations dictionary.
        """
        return set(self.annotations.keys())

    ##########################################################################
    # 2.3) Annotation logic
    ##########################################################################
    def load_current_clip(self):
        if self.current_index >= len(self.clip_paths):
            self._show_done_message()
            return

        rel_path = self.clip_paths[self.current_index]

        # SIMPLEST APPROACH: Use the rel_path directly with the clip_dir
        # Since we're now only collecting files from the clips directory
        # and using relative paths from the base directory
        clip_path = os.path.join(self.clip_dir, rel_path)

        # Normalize the path
        clip_path = os.path.normpath(clip_path)

        # Update progress display
        self.progress_label.setText(f"Clip {self.current_index + 1} of {len(self.clip_paths)}")
        self.progress_bar.setValue(self.ann_count)

        # Enable/disable prev button based on position
        self.prev_button.setEnabled(self.current_index > 0)

        # Progress print
        print(f"\n--- Now annotating clip {self.current_index + 1} of {len(self.clip_paths)} ---")
        print(f"File: {rel_path}")
        print(f"Resolved path: {clip_path}")
        print(f"Already annotated so far: {self.ann_count} of {len(self.clip_paths)}\n")

        # Reset
        self.has_watched_clip = False
        self.chosen_class = None
        self.next_button.setEnabled(False)
        self.is_quarter_speed = False
        self.speed_toggle_button.setText("1/4 Speed (p)")
        self.is_paused = False
        self.pause_button.setText("Pause (space)")
        self.position_slider.setVisible(False)
        self.position_slider.setEnabled(False)

        # Disable class buttons until clip is played through once
        for btn in self.class_buttons:
            btn.setEnabled(False)
            btn.setChecked(False)  # Uncheck all each time we load a new clip

        # If we have a previous annotation for this clip, pre-select it
        if rel_path in self.annotations:
            self.chosen_class = self.annotations[rel_path]
            self.has_watched_clip = True  # Consider it as already watched

        # First stop the player completely
        self.player.stop()

        # Set up playback with explicit file check
        if not os.path.exists(clip_path):
            QMessageBox.warning(self, "File Error", f"Cannot find clip file: {clip_path}\n\nCSV path: {rel_path}")
            print(f"File not found at: {clip_path}")
            return

        print(f"Loading clip: {clip_path}")
        self.player.setPlaybackRate(0.5)
        self.player.setSource(QUrl.fromLocalFile(clip_path))

        # Add a small delay before playing to ensure proper loading
        QTimer.singleShot(100, self.player.play)

        self.start_time = time.perf_counter()

        # If we have a previous annotation, select the corresponding button
        if self.chosen_class is not None:
            # Enable all buttons since we've "watched" it
            for btn in self.class_buttons:
                btn.setEnabled(True)

            # Check the previously selected class button
            if self.chosen_class in self.class_button_map:
                self.class_button_map[self.chosen_class].setChecked(True)
                self.next_button.setEnabled(True)

    @Slot()
    def on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            if not self.is_paused:
                if not self.has_watched_clip:
                    self.has_watched_clip = True
                    for btn in self.class_buttons:
                        btn.setEnabled(True)
                    if self.chosen_class is not None:
                        self.next_button.setEnabled(True)
                # Loop the video
                self.player.setPosition(0)
                self.player.play()

    @Slot()
    def on_class_chosen(self, class_name):
        """
        Highlight the chosen class button and enable Next if user has watched.
        """
        if not self.has_watched_clip:
            return

        # Uncheck all other buttons first
        for btn in self.class_buttons:
            btn.setChecked(False)

        # Check the clicked/shortcutted button
        chosen_btn = self.class_button_map[class_name]
        chosen_btn.setChecked(True)

        self.chosen_class = class_name
        self.next_button.setEnabled(True)

        # Store annotation temporarily (will be written to CSV on next/prev)
        rel_path = self.clip_paths[self.current_index]
        self.annotations[rel_path] = class_name

    @Slot()
    def on_prev_clicked(self):
        """
        Go to the previous clip and save current annotation.
        """
        if self.current_index <= 0:
            return

        # Save current annotation if we have one
        if self.chosen_class is not None:
            self._save_current_annotation()

        # Move to previous clip
        self.current_index -= 1
        self.player.stop()  # Ensure player is fully stopped
        QTimer.singleShot(50, self.load_current_clip)  # Small delay before loading new clip

    @Slot()
    def on_next_clicked(self):
        if self.current_index >= len(self.clip_paths):
            self._show_done_message()
            return
        if not self.chosen_class:
            QMessageBox.warning(self, "No Class Selected", "Please select a class before Next.")
            return

        # Save annotation
        self._save_current_annotation()

        self.current_index += 1
        self.player.stop()  # Ensure player is fully stopped

        if self.current_index >= len(self.clip_paths):
            self._show_done_message()
        else:
            QTimer.singleShot(50, self.load_current_clip)  # Small delay before loading new clip

    def _save_current_annotation(self):
        """
        Save the current annotation to the CSV file.
        """
        rel_path = self.clip_paths[self.current_index]
        end_time = time.perf_counter()
        time_spent = end_time - self.start_time
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if this clip was already annotated
        already_annotated = rel_path in self._get_annotated_clips()

        # Write annotation row
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["clip_path", "user", "date", "time_spent_s", "class"]
            )
            row = {
                "clip_path": rel_path,
                "user": self.user_name,
                "date": now_str,
                "time_spent_s": f"{time_spent:.3f}",
                "class": self.chosen_class,
            }
            writer.writerow(row)

        # Only increment annotation count if it's a new annotation
        if not already_annotated:
            self.ann_count += 1
            self.progress_bar.setValue(self.ann_count)

        print(f"Annotated clip {self.current_index + 1}. Progress: {self.ann_count}/{len(self.clip_paths)}")

    @Slot()
    def on_go_to_clip(self):
        """
        Jump to a specific clip number entered by the user.
        """
        try:
            # Get the clip number from input (1-based index for user, 0-based for code)
            clip_num = int(self.go_to_input.text())
            if clip_num < 1 or clip_num > len(self.clip_paths):
                QMessageBox.warning(self, "Invalid Clip Number",
                                    f"Please enter a number between 1 and {len(self.clip_paths)}.")
                return

            # Save current annotation if we have one
            if self.chosen_class is not None:
                self._save_current_annotation()

            # Move to the specified clip (converting from 1-based to 0-based)
            self.current_index = clip_num - 1
            self.player.stop()  # Ensure player is fully stopped

            # Use a timer to give the player time to fully stop
            QTimer.singleShot(50, self.load_current_clip)

            # Clear the input field
            self.go_to_input.clear()

        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

    @Slot()
    def on_go_to_last_annotated(self):
        """
        Jump to the last annotated clip (highest index in annotated clips).
        """
        if not self.annotations:
            QMessageBox.information(self, "No Annotations", "No annotated clips found.")
            return

        # Save current annotation if we have one
        if self.chosen_class is not None:
            self._save_current_annotation()

        # Find the highest index of an annotated clip by scanning from the end
        last_annotated_index = -1

        # Iterate through clip_paths in reverse order to find the last annotated clip
        for i in range(len(self.clip_paths) - 1, -1, -1):
            if self.clip_paths[i] in self.annotations:
                last_annotated_index = i
                break

        print(f"Found last annotated clip at index: {last_annotated_index}")

        if last_annotated_index >= 0:
            # Navigate to that clip
            self.current_index = last_annotated_index
            self.player.stop()  # Ensure player is fully stopped
            QTimer.singleShot(50, self.load_current_clip)
        else:
            QMessageBox.information(self, "Navigation Error",
                                    "Could not determine the last annotated clip. Please try again.")

    def _show_done_message(self):
        QMessageBox.information(self, "Done", "All clips have been annotated!")
        self.close()

    ##########################################################################
    # 2.4) Speed and Pause
    ##########################################################################
    def toggle_speed(self):
        """
        Toggled by hotkey 'p' or button click.
        Switch between 0.25× and 0.5× speed.
        """
        if not self.is_quarter_speed:
            self.player.setPlaybackRate(0.25)
            self.speed_toggle_button.setText("1/2 Speed (p)")
            self.is_quarter_speed = True
        else:
            self.player.setPlaybackRate(0.5)
            self.speed_toggle_button.setText("1/4 Speed (p)")
            self.is_quarter_speed = False

    def toggle_pause(self):
        if not self.is_paused:
            self.player.pause()
            self.pause_button.setText("Play (space)")
            self.is_paused = True
            self.position_slider.setVisible(True)
            self.position_slider.setEnabled(True)
        else:
            self.player.play()
            self.pause_button.setText("Pause (space)")
            self.is_paused = False
            self.position_slider.setVisible(False)
            self.position_slider.setEnabled(False)

    ##########################################################################
    # 2.5) Slider
    ##########################################################################
    def on_duration_changed(self, duration):
        if duration <= 0:
            self.position_slider.setRange(0, 0)
        else:
            self.position_slider.setRange(0, duration)

    def on_position_changed(self, position):
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)

    def on_slider_moved(self, new_position):
        if self.is_paused:
            self.player.setPosition(new_position)

    ##########################################################################
    # 2.6) Close / Quit
    ##########################################################################
    def close_app(self):
        # Save current annotation if we have one
        if self.chosen_class is not None:
            self._save_current_annotation()

        self.player.stop()
        self.close()
        QApplication.instance().quit()


##############################################################################
# 3) Main
##############################################################################
def main():
    app = QApplication(sys.argv)
    window = AnnotationWindow()
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()