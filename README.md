# Map Cleaner

A small tool for cleaning occupancy grid maps (`.pgm` + `.yaml`) using segmentation from MobileSAM. It is designed to take a noisy or imperfect map, extract the usable region (such as a track or navigable space), and convert it into a clean occupancy grid with consistent values.

## What it does

- Loads maps (`.pgm` + `.yaml`)
- Uses MobileSAM for segmentation (automatic or point-based)
- Supports **multiple point prompts** for more precise control
- Cleans and smooths the segmented region
- Saves the cleaned map

---

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/459f1fb1-d7a0-4de2-a538-d54aacd605d3" />
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/01489a69-eabd-4fc6-892e-4ae987062230" />






