# Metal Assembly Inspector

A small desktop tool that compares two photos of a metal assembly and
automatically highlights every visual difference, so a technician can spot a
mis-weld, a missing bracket, or a missing bolt in seconds instead of
inspecting the part by eye.

Select **Image A** (a known-good reference assembly) and **Image B** (the
assembly being inspected), click **Compare**, and you get one composite image:
Image A on the left for reference, Image B on the right with every difference
region boxed and highlighted in red.

---

## 1. Installation

Requires **Python 3.9+**.

```bash
pip install -r requirements.txt
```

`tkinter` (the GUI toolkit) ships with most standard Python installers on
Windows and macOS. On some Linux distributions it's a separate system
package:

```bash
sudo apt install python3-tk        # Debian/Ubuntu
sudo dnf install python3-tkinter   # Fedora
```

If you only need the command-line mode (see below), `tkinter` is not
required at all.

## 2. Running the app

**GUI:**
```bash
python app.py
```
1. Click **Browse Image A (reference)...** and pick the known-good photo.
2. Click **Browse Image B (inspected)...** and pick the photo to check.
3. Click **Compare**. The result appears in a few seconds.
4. Click **Save Result...** to export the composite as a JPEG/PNG.

**Command line** (useful for batch testing or scripting into a QA pipeline):
```bash
python app.py --cli path/to/imageA.jpg path/to/imageB.jpg path/to/result.jpg
```
This prints the similarity score, alignment confidence, and every difference
region's coordinates, and saves the same composite image the GUI would show.

## 3. Adding your own test images

Drop any two JPEGs into the `sample_images/` folder and point the app at
them. The `sample_images/` and `sample_output/` folders included with this
submission show three worked examples (see section 5) so you can confirm the
tool is working correctly before trying your own shop photos.

## 4. How it works

`compare_core.py` contains the whole algorithm, in five steps:

1. **Alignment** - ORB feature matching + RANSAC homography warps Image B
   onto Image A's perspective, so the two photos line up even if the camera
   wasn't in the exact same spot for both shots.
2. **Lighting/reflection normalization** - both images are converted to
   grayscale, run through CLAHE (adaptive contrast normalization) and a
   blur pass, and any blown-out specular glare (common on polished stainless
   steel) is excluded from the comparison rather than flagged as a defect.
3. **Structural difference** - `scikit-image`'s SSIM (structural similarity)
   gives a difference map that is naturally robust to uniform
   brightness/contrast shifts but sensitive to real structural changes.
4. **Cleanup** - morphological opening/closing removes single-pixel noise
   and sub-pixel alignment jitter along sharp edges, while keeping real,
   solid defect regions intact.
5. **Region filtering + drawing** - remaining regions are filtered by a
   minimum-area threshold (to ignore texture-level noise) and drawn as
   numbered, semi-transparent red boxes on Image B.

### Tuning knobs

Everything adjustable lives in the `CONFIG` dictionary at the top of
`compare_core.py`, with a comment on what each value does. The three you're
most likely to touch:

| Setting | Effect |
|---|---|
| `MIN_AREA_FRACTION` | Lower = catches smaller defects, but more false positives from texture/reflections. Raise if small noise is getting flagged. |
| `DIFF_BINARY_THRESH` | Lower = more sensitive to subtle differences. Raise to reduce false positives. |
| `SPECULAR_BRIGHTNESS` | Lower this if glare on very shiny stainless steel is still slipping through as a false difference. |

### A note on "millimeter-scale" defects

The tool works in image pixels, not physical units - it has no way to know
how many millimeters one pixel represents unless it's told. In practice this
is rarely a problem: at a typical inspection distance (assembly filling most
of the frame, similar to the sample photos), a few millimeters on the part
is already many pixels wide, well above the noise floor the filtering
removes. If you need a hard physical threshold, the cleanest way is to
place a fixed reference dimension (e.g. a ruler or a known bolt-head width)
in frame once, measure its pixel size, and use that ratio to convert
`MIN_AREA_FRACTION` into a real-world size for your specific camera setup.

### A note on camera viewpoint

For reliable pixel-level comparison, Image A and Image B should be taken
from a similar angle and distance (e.g. a fixed phone holder or repeatable
spot on the shop floor) - this is standard practice for visual QA stations
and is what makes catching small defects possible at all. If the two photos
are taken from very different angles, a single geometric alignment can't
fully reconcile the 3D parallax, and the tool will say so: it reports an
**alignment confidence** of `ok`, `low`, or `none`, and adds an on-image
warning banner instead of silently producing a wall of false positives.

## 5. Included worked examples

`sample_output/` contains three results generated from the images in
`sample_images/`, showing both ends of what the tool does:

- **`flange_synthetic_missing_bolt.jpg`** - Image A is a reference photo of a
  bolted flange bracket; Image B is the same photo with one bolt digitally
  removed and the whole image brightened/noised to simulate a different
  lighting/exposure. Alignment confidence is `ok`, similarity score is
  0.97, and exactly one region is flagged - the missing bolt - with the
  simulated global lighting change correctly ignored.
- **`flange_mismatched_viewpoint.jpg`** and **`elbow_mismatched_viewpoint.jpg`**
  - two genuinely different shop photos of similar assemblies, taken from
  different angles/backgrounds. Alignment confidence correctly drops to
  `low` and the caution banner appears, rather than the tool guessing.

Regenerate any of these yourself with, e.g.:
```bash
python app.py --cli sample_images/flange_A.jpg sample_images/flange_A_defect_sim.jpg out.jpg
```

## 6. Packaging as a standalone executable (optional)

If you'd like a single .exe to hand to shop-floor PCs without a Python
install:
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name MetalInspector app.py
```
The executable will be created in `dist/`.

## 7. Files in this submission

```
app.py               GUI + CLI entry point
compare_core.py       Core comparison engine (alignment, diffing, drawing)
requirements.txt      Python dependencies
README.md             This file
sample_images/        Test images used to produce the worked examples
sample_output/        Pre-generated results from those test images
```
