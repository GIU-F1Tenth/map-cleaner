# Map Cleaner

A small tool for cleaning occupancy grid maps with MobileSAM. It loads a map image plus YAML metadata, uses point-prompted segmentation, and saves a cleaned occupancy map.

## Requirements

- Python 3.10 or 3.11 recommended
- Git, because MobileSAM is installed from GitHub
- Internet access on first setup
- A map YAML file whose `image` field points to the map image file

## Installation

Clone the repository:

```powershell
git clone https://github.com/GIU-F1Tenth/map-cleaner.git
cd map-cleaner
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists the direct runtime dependencies. Pip resolves the
extra packages they need automatically.

The MobileSAM Python package is installed by `requirements.txt`, but the
checkpoint file is not stored in the repository. The first segmentation run
downloads `mobile_sam.pt` into `src/models/`.

PyTorch installation can vary by platform and GPU/CUDA setup. If you need a
specific CUDA build, install the matching PyTorch packages first, then run:

```powershell
python -m pip install -r requirements.txt
```

Run the app:

```powershell
python src/main.py
```

## Linux/macOS Setup

```bash
git clone https://github.com/GIU-F1Tenth/map-cleaner.git
cd map-cleaner
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/main.py
```

## Map Files

Place maps in `src/maps/` or choose them from anywhere using the file picker. A typical YAML file should include an `image` entry:

```yaml
image: my_map.pgm
resolution: 0.05
origin: [0.0, 0.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

The cleaned map is saved as a single grayscale PNG.

## GUI Functionality

The app is split into a control panel on the left and a map preview on the right.

The example screenshots are useful references for the main workflow: the
high-thickness example shows what the wall thickness slider does, the point
example shows SAM prompt placement, the low-thickness/low-smoothing example
shows a thin raw-looking wall, and the high-smoothing example shows a cleaner
less pixelated outline.

### Map File

Use `Browse .yaml...` to load a map YAML file. The YAML file must contain an `image` field that points to the map image. Once loaded, the selected filename appears under `Map file`, and the map appears in the preview.

### Prompt Points

Prompt points tell MobileSAM which parts of the image should belong to the segmented region.

- Select `Points` mode.
- Click inside the preview to add points.
- Each point appears as a colored marker on the map and as an entry in the `Prompt points` list.
- Use the red `X` beside a point to delete that point.
- Use `Clear all points` to remove every point.

The point screenshot demonstrates this stage: colored numbered points are placed around the drivable/map region so SAM has multiple hints about what to segment.

### Canvas Modes

The `Canvas mode` selector controls what clicking or dragging on the preview does.

Painting before a run, or on the original panel in `3 images` mode, creates
helper edits that are used while processing the map. Painting on the cleaned
result after a run directly changes the final output that will be saved.

`Points` adds MobileSAM prompt points.

`Black` paints black occupied cells. Before running segmentation, painting black on the original image is useful for manually closing broken borders or gaps so the cleaner can interpret the region correctly. These helper walls are used for processing and are removed from the final cleaned output. After running segmentation, painting black on the cleaned output directly edits the final map.

`Erase line` erases black lines that were manually painted with the `Black` brush. It does not erase generated walls or turn arbitrary output pixels gray.

`Gray` paints unknown/outside map cells. Use it to remove regions from the cleaned output by turning them back into the map background value.

`Green` paints free/internal track space in the preview. In the saved occupancy map this is stored as free space.

`White` also paints free space, but keeps that region visually white in the preview image. In the saved occupancy map it uses the same free-space value as `Green`.

### Brush Size

`Brush size (px)` controls the width of manual painting and erasing. The brush cursor shown in the preview follows the selected brush size, so the circle indicates the approximate area affected by a stroke.

The clear buttons reset manual edits by type:

- `Clear painted walls` removes manually painted black helper walls.
- `Clear gray edits` removes manually painted gray edits.
- `Clear green/white edits` removes manually painted free-space edits.

### Wall Thickness

`Wall thickness (px)` controls the thickness of the black occupied boundary drawn around the cleaned free-space region.

Low values create thin walls. High values create heavier boundaries. The high-thickness screenshot shows this clearly: the black wall around the cleaned green region becomes much wider.

#### High Wall Thickness Example

![High wall thickness example](<src/imgs/High Thickness.png>)

#### Low Wall Thickness Example

![Low wall thickness example](<src/imgs/Low Thickness.png>)

### Wall Smoothing

`Wall smoothing (sigma)` controls how much the generated wall outline is smoothed.

Low smoothing keeps the wall close to the raw pixel boundary, which can look stair-stepped or jagged. The low-thickness, low-smoothness screenshot shows a thinner wall with more visible pixelation.

High smoothing makes the wall outline less pixelated by smoothing the contour used to draw the black boundary. The high-smoothness screenshot shows the same general map shape with a cleaner, less jagged edge. Smoothing is intended to affect wall appearance, not to redefine the whole segmented map body.

#### Low Wall Smoothing Example

![Low wall smoothing example](<src/imgs/Low Smoothness.png>)

#### High Wall Smoothing Example

![High wall smoothing example](<src/imgs/Low Thickness.png>)

### Min Noise Dot Area

`Min noise dot area (px)` controls how aggressively small holes or isolated noise dots are cleaned during processing.

Lower values preserve more small details. Higher values remove or fill larger small artifacts. This is useful when the original map contains scattered dots, speckles, or tiny unwanted holes.

### Run

`Run` sends the loaded map and prompt points through MobileSAM, then converts the resulting mask into a cleaned occupancy map. At least one prompt point is required.

The first run may take longer because the MobileSAM weights are downloaded automatically if they are not already present in `src/models/`.

### Save Map

`Save map` writes only the cleaned grayscale PNG. It does not create a YAML file or a preview image.

The saved occupancy map uses standard grayscale occupancy values: free space, occupied walls, and unknown/background.

### Preview Controls

The preview toolbar controls how the output is displayed.

`Result only` shows only the cleaned output. This is the default view.

`3 images` shows the original map, the SAM mask, and the cleaned output side by side. This is useful for comparing what SAM selected against the final cleaned map.

`-` zooms out.

`+` zooms in.

`Fit` resets the preview to fit the window.

The percentage label shows the current zoom level. The coordinate readout in the top-right shows the map pixel under the cursor.

You can also use the mouse wheel to zoom. Right-drag or middle-drag pans the preview.

### Status And Statistics

After processing, the status area shows the percentage of the output map that is free, occupied, and unknown. It also reports current actions such as missing prompt points, completed runs, preview mode changes, or saved files.

## Troubleshooting

If MobileSAM fails to install, confirm Git is available:

```powershell
git --version
```

If the model download fails, check internet access and rerun the app. The model file should appear at:

```text
src/models/mobile_sam.pt
```
