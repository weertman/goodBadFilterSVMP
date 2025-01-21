import sys
import os
import random
import csv
import time
from datetime import datetime

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLineEdit, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QSlider
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

        # "Next" button (triggered by Tab)
        self.next_button = QPushButton("Next (Tab)")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.on_next_clicked)

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

        # Class buttons row
        class_buttons_layout = QHBoxLayout()
        for btn in self.class_buttons:
            class_buttons_layout.addWidget(btn)
        annotation_layout.addLayout(class_buttons_layout)

        # Bottom row: speed toggle + pause + next + close
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.speed_toggle_button)
        bottom_layout.addWidget(self.pause_button)
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

        self.csv_path = os.path.join(self.clip_dir, f"annotations_{self.user_name}.csv")

        # Gather .mp4 files
        all_clips = []
        for root, dirs, files in os.walk(self.clip_dir):
            for fname in files:
                if fname.lower().endswith(".mp4"):
                    rel_path = os.path.relpath(os.path.join(root, fname), self.clip_dir)
                    all_clips.append(rel_path)

        if not all_clips:
            QMessageBox.information(self, "No Clips", "No .mp4 clips found in the directory.")
            return

        # Load or create CSV
        if os.path.exists(self.csv_path):
            existing_clips = set()
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_clips.add(row["clip_path"])
            new_clips = [c for c in all_clips if c not in existing_clips]
            random.shuffle(new_clips)
            self.clip_paths = list(existing_clips) + new_clips
        else:
            self.clip_paths = list(all_clips)
            random.shuffle(self.clip_paths)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["clip_path", "user", "date", "time_spent_s", "class"]
                )
                writer.writeheader()

        # Count how many are annotated
        annotated_set = self._get_annotated_clips()
        self.ann_count = len(annotated_set)

        # Switch to annotation mode
        self.setup_widget.setVisible(False)
        self.annotation_widget.setVisible(True)

        # Make sure none of the annotation widgets can receive tab-focus
        # so that our keyPressEvent will see Tab.
        for child in self.annotation_widget.findChildren(QWidget):
            child.setFocusPolicy(Qt.NoFocus)

        # Give the main window focus so it receives key events
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # Go to the first unannotated clip
        for i, path in enumerate(self.clip_paths):
            if path not in annotated_set:
                self.current_index = i
                break
        else:
            self._show_done_message()
            return

        self.load_current_clip()

    def _get_annotated_clips(self):
        annotated = set()
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["class"]:
                        annotated.add(row["clip_path"])
        return annotated

    ##########################################################################
    # 2.3) Annotation logic
    ##########################################################################
    def load_current_clip(self):
        if self.current_index >= len(self.clip_paths):
            self._show_done_message()
            return

        rel_path = self.clip_paths[self.current_index]
        clip_path = os.path.join(self.clip_dir, rel_path)

        # Progress print
        print(f"\n--- Now annotating clip {self.current_index+1} of {len(self.clip_paths)} ---")
        print(f"File: {rel_path}")
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

        # Set up playback
        self.player.setPlaybackRate(0.5)
        self.player.setSource(QUrl.fromLocalFile(clip_path))
        self.player.play()

        self.start_time = time.perf_counter()

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

    @Slot()
    def on_next_clicked(self):
        if self.current_index >= len(self.clip_paths):
            self._show_done_message()
            return
        if not self.chosen_class:
            QMessageBox.warning(self, "No Class Selected", "Please select a class before Next.")
            return

        self.player.stop()
        rel_path = self.clip_paths[self.current_index]
        end_time = time.perf_counter()
        time_spent = end_time - self.start_time
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

        self.ann_count += 1
        print(f"Annotated clip {self.current_index + 1}. Progress: {self.ann_count}/{len(self.clip_paths)}")

        self.current_index += 1
        if self.current_index >= len(self.clip_paths):
            self._show_done_message()
        else:
            self.load_current_clip()

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
