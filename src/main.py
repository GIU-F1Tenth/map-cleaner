"""
main.py
Entry point for the RoboRacer Map Cleaner GUI.

Run:
    python main.py

Requirements:
    pip install ultralytics opencv-python numpy pyyaml PyQt5
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui import MapCleanerWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MapCleanerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()