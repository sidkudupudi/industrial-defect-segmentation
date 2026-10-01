"""
Compose one image per C++ demo case: MVTec ground-truth outline (left) next to the C++/TensorRT output (right),
and write a summary table of what the C++ app reported.

The C++ app draws each instance mask by adding a red tint (addWeighted with a pure-red layer), so the predicted mask is
recovered from the output image as the pixels whose red channel rose clearly above the input. It is then scored against
the MVTec ground-truth mask:
    IoU >= 0.5                                          -> "located"
    IoU < 0.5, mask mostly on the defect                -> "partly covered"
    IoU < 0.5, covers >= half of the defect but spills  -> "over-segmented"
    any other detection                                 -> "wrong area"
    no instance above the app's 0.5 threshold     -> "missed"
Defect-free parts are "false alarm" if anything is detected, otherwise "correct: no detection".

Usage:
    python src/render_cpp_demo.py --mvtec <MVTec AD root> --cases cases.csv --raw <run_demo.sh out_dir> --out results/cpp_demo
cases.csv columns: case, category, defect, image   (defect = "good" for defect-free parts)
"""
import argparse, re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CLASSES = ["cable", "screw", "metal_nut", "toothbrush", "transistor", "zipper", "bottle", "capsule",
           "wood", "hazelnut", "carpet", "leather", "grid", "tile", "pill"]           # mvtec.yaml order
PANEL, HEADER = 760, 64
INK, MUTED, YELLOW = (20, 20, 20), (110, 108, 104), (0, 215, 255)


def panel(img, title, subtitle):
    img = cv2.resize(img, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    head = np.full((HEADER, PANEL, 3), 251, np.uint8)
    cv2.putText(head, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, INK, 2, cv2.LINE_AA)
    cv2.putText(head, subtitle, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, MUTED, 1, cv2.LINE_AA)
    return np.vstack([head, img])


def recover_mask(src, out):
    """Pixels the C++ app tinted red (its mask overlay), excluding the green box lines and JPEG noise."""
    d = out.astype(int) - src.astype(int)
    red = (d[..., 2] > 40) & (d[..., 2] - np.abs(d[..., 1]) > 30)
    return cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in pd.read_csv(a.cases).itertuples():
        src = cv2.imread(str(a.mvtec / c.image), cv2.IMREAD_COLOR)
        out = cv2.imread(str(a.raw / f"{c.case}.jpg"))
        log = (a.raw / f"{c.case}.log").read_text()
        dets = [(CLASSES[int(k)], float(v)) for k, v in re.findall(r"Class ID: (\d+) \| Confidence: ([\d.]+)", log)]
        pred = recover_mask(src, out)
        gt, iou, cover = src.copy(), None, None
        if c.defect != "good":
            mask_path = a.mvtec / c.category / "ground_truth" / c.defect / (Path(c.image).stem + "_mask.png")
            g = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
            iou = float((pred & g).sum() / max((pred | g).sum(), 1))
            cover = float((pred & g).sum() / max(g.sum(), 1))
            contours, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt, contours, -1, YELLOW, max(2, src.shape[0] // 250), cv2.LINE_AA)
            left_title, left_sub = f"{c.category} / {c.defect.replace('_', ' ')}", "input with ground-truth defect outline"
        else:
            left_title, left_sub = f"{c.category} / defect-free part", "input (no defect)"

        if c.defect == "good":
            verdict = "false alarm" if dets else "correct: no detection"
        elif not dets:
            verdict = "missed"
        else:
            on_defect = float((pred & g).sum() / max(pred.sum(), 1))       # share of the predicted mask lying on the defect
            if iou >= 0.5:
                verdict = "located"
            elif on_defect >= 0.5:
                verdict = "partly covered"
            elif cover >= 0.5:
                verdict = "over-segmented"
            else:
                verdict = "wrong area"
        found = "; ".join(f"{k} {v:.2f}" for k, v in dets) if dets else "no instance above confidence 0.5"
        shown_iou = np.floor(iou * 100) / 100 if iou is not None else None          # truncate so 0.496 never reads as 0.50
        right_sub = f"{found}  |  mask IoU {shown_iou:.2f}: {verdict}" if (iou is not None and dets) else f"{found}  |  {verdict}"

        fig = np.hstack([panel(gt, left_title, left_sub), np.full((PANEL + HEADER, 12, 3), 251, np.uint8),
                         panel(out, "C++ / TensorRT output", right_sub)])
        cv2.imwrite(str(a.out / f"{c.case}.jpg"), fig, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(dict(case=c.case, category=c.category, defect=c.defect, detections=len(dets),
                         predicted_classes=", ".join(k for k, _ in dets),
                         max_confidence=max((v for _, v in dets), default=None),
                         category_correct=(c.category in [k for k, _ in dets]) if c.defect != "good" else None,
                         mask_iou=iou, gt_covered=cover, verdict=verdict))
    table = pd.DataFrame(rows)
    table.to_csv(a.out / "cpp_demo_summary.csv", index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
