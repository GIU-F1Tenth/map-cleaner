# Map Cleaner

A small tool for cleaning occupancy grid maps (`.pgm` + `.yaml`) using segmentation from MobileSAM. It is designed to take a noisy or imperfect map, extract the usable region (such as a track or navigable space), and convert it into a clean occupancy grid with consistent values.

## What it does

- Loads maps (`.pgm` + `.yaml`)
- Uses MobileSAM for segmentation (automatic or point-based)
- Supports **multiple point prompts** for more precise control
- Cleans and smooths the segmented region
- Saves the cleaned map

---

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/19468686-3f7a-4c00-9d09-afed146d2a93" />
<img width="1344" height="524" alt="image" src="https://github.com/user-attachments/assets/6ada8fa5-2ac8-4967-ba87-77a2bebc3341" />





