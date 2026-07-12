"""
compare_core.py
----------------
Core image-comparison engine for the Metal Assembly Inspector.

Given two JPEG photos of a (supposedly) identical metal assembly, this module:
  1. Aligns Image B onto Image A's perspective (handles the fact that the two
     shop photos are almost never taken from the exact same tripod position).
  2. Normalizes lighting/reflections so shiny stainless steel doesn't trigger
     false positives.
  3. Computes a structural difference map between the two aligned images.
  4. Cleans up the difference map (removes single-pixel noise / specular
     glints, keeps real regions).
  5. Draws bounding boxes + a semi-transparent highlight around each real
     difference on a copy of Image B.
  6. Builds a single side-by-side composite: Image A (reference) on the left,
     annotated Image B on the right.

All tunable knobs are collected in the CONFIG dict at the top of this file so
an in-house developer can adjust them without reading the algorithm itself.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# TUNABLE SETTINGS
# ---------------------------------------------------------------------------
CONFIG = {
    # Work at this max dimension internally for speed & consistency, then
    # scale results back up. 900px is plenty for spotting mm-scale defects
    # on parts that fill most of the frame.
    "WORKING_MAX_DIM": 900,

    # --- Alignment (ORB feature matching + homography) ---
    "ORB_FEATURES": 4000,
    "MATCH_RATIO": 0.75,           # Lowe's ratio test
    "MIN_MATCH_COUNT": 12,         # below this, skip alignment (use as-is)
    "RANSAC_REPROJ_THRESHOLD": 5.0,

    # --- Lighting / reflection normalization ---
    "CLAHE_CLIP_LIMIT": 2.5,
    "CLAHE_TILE_GRID": (8, 8),
    "BLUR_KERNEL": (9, 9),           # reduces sensor noise & sub-pixel edge jitter
    "SPECULAR_BRIGHTNESS": 245,     # pixels this bright are treated as glare
                                     # and excluded from the diff calculation

    # --- Difference detection ---
    "SSIM_WINDOW": 7,
    "DIFF_BINARY_THRESH": 25,       # 0-255 on the inverted-SSIM map
    "MORPH_KERNEL": (7, 7),
    "MORPH_OPEN_ITER": 2,           # removes small noise specks / edge jitter
    "MORPH_CLOSE_ITER": 2,          # joins broken pieces of one real defect

    # --- Region filtering ---
    # Minimum defect area as a FRACTION of image area. Real components /
    # missing brackets / mis-welds are rarely smaller than ~0.05% of frame.
    # Tune this down if you need to catch smaller defects; tune it up if
    # you get too many false positives from texture/reflections.
    "MIN_AREA_FRACTION": 0.0009,

    # --- Drawing ---
    "BOX_COLOR": (0, 0, 255),        # red (BGR) bounding box
    "FILL_COLOR": (0, 0, 255),       # red translucent fill
    "FILL_ALPHA": 0.35,
    "BOX_THICKNESS": 3,
    "LABEL_FONT_SCALE": 0.6,

    "DIVIDER_WIDTH": 6,
    "DIVIDER_COLOR": (40, 40, 40),
    "HEADER_HEIGHT": 40,
    "HEADER_BG": (30, 30, 30),
    "HEADER_TEXT_COLOR": (255, 255, 255),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_to_max_dim(img, max_dim):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img.copy(), 1.0
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale


def _to_gray_normalized(img_bgr, cfg):
    """Grayscale + CLAHE + blur, to make stainless-steel reflections and
    minor lighting shifts stop mattering for the diff calculation."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=cfg["CLAHE_CLIP_LIMIT"],
                             tileGridSize=cfg["CLAHE_TILE_GRID"])
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, cfg["BLUR_KERNEL"], 0)
    return gray


def align_images(img_a_bgr, img_b_bgr, cfg=CONFIG):
    """
    Aligns img_b onto img_a's frame using ORB features + homography.
    Returns (aligned_b, confidence) where confidence is one of:
        "ok"        - enough inlier matches found, homography is trustworthy
        "low"       - homography computed but weakly supported (few inliers /
                      low inlier ratio) -- likely the two photos were taken
                      from meaningfully different angles/distances
        "none"      - not enough matches at all; returned image is just
                      resized to match, not geometrically aligned
    """
    h, w = img_a_bgr.shape[:2]

    gray_a = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=cfg["ORB_FEATURES"])
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return cv2.resize(img_b_bgr, (w, h)), "none"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des_b, des_a, k=2)

    good = []
    for m, n in [p for p in raw_matches if len(p) == 2]:
        if m.distance < cfg["MATCH_RATIO"] * n.distance:
            good.append(m)

    if len(good) < cfg["MIN_MATCH_COUNT"]:
        return cv2.resize(img_b_bgr, (w, h)), "none"

    src_pts = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                         cfg["RANSAC_REPROJ_THRESHOLD"])
    if H is None:
        return cv2.resize(img_b_bgr, (w, h)), "none"

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = n_inliers / max(len(good), 1)

    aligned_b = cv2.warpPerspective(img_b_bgr, H, (w, h))

    if n_inliers >= cfg["MIN_MATCH_COUNT"] and inlier_ratio >= 0.5:
        confidence = "ok"
    else:
        confidence = "low"

    return aligned_b, confidence


def compute_difference_mask(img_a_bgr, img_b_bgr, cfg=CONFIG):
    """
    Returns a clean binary mask (uint8, 0/255) the same size as the inputs,
    where 255 = a real visual difference between the two assemblies.
    """
    gray_a = _to_gray_normalized(img_a_bgr, cfg)
    gray_b = _to_gray_normalized(img_b_bgr, cfg)

    # Structural similarity map: robust to uniform brightness/contrast shift,
    # sensitive to actual structural change (missing part, shifted weld...).
    score, diff = ssim(gray_a, gray_b, win_size=cfg["SSIM_WINDOW"], full=True)
    diff = (1.0 - diff)
    diff = np.clip(diff * 255, 0, 255).astype(np.uint8)

    _, mask = cv2.threshold(diff, cfg["DIFF_BINARY_THRESH"], 255,
                             cv2.THRESH_BINARY)

    # Exclude blown-out specular glare in EITHER image from counting as a
    # difference -- glare moves around with camera angle and is not a real
    # manufacturing defect.
    glare_a = gray_a > cfg["SPECULAR_BRIGHTNESS"]
    glare_b = gray_b > cfg["SPECULAR_BRIGHTNESS"]
    glare = np.logical_or(glare_a, glare_b).astype(np.uint8) * 255
    mask[glare > 0] = 0

    # Morphological cleanup: opening kills lone-pixel noise, closing
    # reconnects a single real defect that got fragmented.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg["MORPH_KERNEL"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                             iterations=cfg["MORPH_OPEN_ITER"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                             iterations=cfg["MORPH_CLOSE_ITER"])

    return mask, score


def find_diff_boxes(mask, cfg=CONFIG):
    """Returns a list of (x, y, w, h) bounding boxes for regions in `mask`
    large enough to be a real defect, filtered by MIN_AREA_FRACTION."""
    h, w = mask.shape[:2]
    min_area = cfg["MIN_AREA_FRACTION"] * (h * w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        boxes.append(cv2.boundingRect(c))
    return boxes


def annotate_image(img_bgr, boxes, cfg=CONFIG):
    """Draws translucent fill + bounding box + numbered label for each
    difference region on a copy of img_bgr."""
    out = img_bgr.copy()
    overlay = img_bgr.copy()

    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), cfg["FILL_COLOR"], -1)
    out = cv2.addWeighted(overlay, cfg["FILL_ALPHA"], out,
                           1 - cfg["FILL_ALPHA"], 0)

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), cfg["BOX_COLOR"],
                       cfg["BOX_THICKNESS"])
        label = f"#{i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       cfg["LABEL_FONT_SCALE"], 2)
        ly = max(y - 8, th + 4)
        cv2.rectangle(out, (x, ly - th - 6), (x + tw + 8, ly + 2),
                       cfg["BOX_COLOR"], -1)
        cv2.putText(out, label, (x + 4, ly - 4), cv2.FONT_HERSHEY_SIMPLEX,
                     cfg["LABEL_FONT_SCALE"], (255, 255, 255), 2,
                     cv2.LINE_AA)
    return out


def _add_header(img, text, cfg=CONFIG):
    h, w = img.shape[:2]
    header = np.full((cfg["HEADER_HEIGHT"], w, 3), cfg["HEADER_BG"],
                      dtype=np.uint8)
    cv2.putText(header, text, (10, cfg["HEADER_HEIGHT"] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, cfg["HEADER_TEXT_COLOR"], 2,
                cv2.LINE_AA)
    return np.vstack([header, img])


def build_side_by_side(img_a_bgr, annotated_b_bgr, cfg=CONFIG, warning=None):
    """Stacks Image A (left, reference) and annotated Image B (right) into
    one composite image with header labels and a divider bar. If `warning`
    is given, a banner is added across the top of the whole composite."""
    h = max(img_a_bgr.shape[0], annotated_b_bgr.shape[0])

    def fit(img):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    left = _add_header(fit(img_a_bgr), "Reference (Image A)", cfg)
    right = _add_header(fit(annotated_b_bgr), "Inspected (Image B) - differences highlighted", cfg)

    divider = np.full((left.shape[0], cfg["DIVIDER_WIDTH"], 3),
                       cfg["DIVIDER_COLOR"], dtype=np.uint8)

    composite = np.hstack([left, divider, right])

    if warning:
        banner_h = 34
        banner = np.full((banner_h, composite.shape[1], 3), (0, 140, 255),
                          dtype=np.uint8)
        cv2.putText(banner, warning, (10, banner_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        composite = np.vstack([banner, composite])

    return composite


def compare(img_a_bgr, img_b_bgr, cfg=CONFIG):
    """
    Full pipeline. Takes two BGR images (as loaded by cv2.imread), returns:
        composite_bgr : the side-by-side result image (includes a warning
                         banner if the two photos appear to be from very
                         different viewpoints)
        boxes         : list of (x, y, w, h) difference regions (in the
                         working/resized coordinate space used internally)
        similarity    : SSIM score 0-1 (1.0 = identical)
        confidence    : "ok" / "low" / "none" -- alignment confidence.
                        "low"/"none" means the two photos were likely not
                        taken from a similar enough camera position for a
                        reliable pixel-level comparison; results should be
                        treated with caution in that case.
    """
    # Work at a consistent resolution for speed + repeatable thresholds.
    work_a, _ = _resize_to_max_dim(img_a_bgr, cfg["WORKING_MAX_DIM"])
    work_b, _ = _resize_to_max_dim(img_b_bgr, cfg["WORKING_MAX_DIM"])
    work_b = cv2.resize(work_b, (work_a.shape[1], work_a.shape[0]))

    aligned_b, confidence = align_images(work_a, work_b, cfg)

    mask, similarity = compute_difference_mask(work_a, aligned_b, cfg)
    boxes = find_diff_boxes(mask, cfg)

    annotated_b = annotate_image(aligned_b, boxes, cfg)

    warning = None
    if confidence == "none":
        warning = ("LOW CONFIDENCE: could not reliably match viewpoints -- "
                    "retake photos from a similar angle/distance for accurate results")
    elif confidence == "low":
        warning = ("CAUTION: photos appear to be from noticeably different "
                    "angles/distances -- verify highlighted regions manually")

    composite = build_side_by_side(work_a, annotated_b, cfg, warning=warning)

    return composite, boxes, similarity, confidence
