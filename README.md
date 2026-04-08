# Map Cleaner

A small tool for cleaning occupancy grid maps (`.pgm` + `.yaml`) using segmentation from MobileSAM. It is designed to take a noisy or imperfect map, extract the usable region (such as a track or navigable space), and convert it into a clean occupancy grid with consistent values.

## What it does

- Loads maps (`.pgm` + `.yaml`)
- Uses MobileSAM for segmentation (automatic or point-based)
- Supports **multiple point prompts** for more precise control
- Cleans and smooths the segmented region
- Converts the result into a 3-value occupancy grid:
  - free: 255  
  - occupied: 0  
  - unknown: 205  
- Displays a side-by-side comparison (original vs cleaned)
- Saves the cleaned map

---





