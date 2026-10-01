"""
Extra evaluation of the trained MVTec defect-segmentation models (run from the project root).

1. Per-category box / mask AP on the MVTec validation split (the same split used during training).
2. False-alarm rate on defect-free parts: MVTec `<category>/test/good` images (467 images the model never saw,
   because training used defective images only). An image counts as a false alarm if the model returns any
   instance above the confidence threshold.
"""
import json, glob, os
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

OUT = Path("extra_eval")
MODELS = {
    "yolo11l-seg_640": ("runs/segment/industrial_pipeline/defect_seg-2/weights/best.pt", 640),
    "yolo11x-seg_1024_tuned": ("runs/segment/industrial_pipeline/mvtec_ultimate_seg-2/weights/best.pt", 1024),
}
CATEGORIES = sorted(Path(p).parent.parent.name for p in glob.glob("*/test/good"))

rows, fa_rows = [], []
for tag, (weights, imgsz) in MODELS.items():
    model = YOLO(weights)
    m = model.val(data="mvtec_yolo/mvtec.yaml", imgsz=imgsz, batch=8, device=0, plots=False, verbose=False,
                  project=str(OUT.resolve() / "val_runs"), name=tag, exist_ok=True)
    names = model.names
    for i, c in enumerate(m.ap_class_index):
        rows.append(dict(model=tag, category=names[int(c)], box_AP50=m.box.ap50[i], box_AP50_95=m.box.ap[i],
                         mask_AP50=m.seg.ap50[i], mask_AP50_95=m.seg.ap[i]))
    rows.append(dict(model=tag, category="ALL", box_AP50=m.box.map50, box_AP50_95=m.box.map,
                     mask_AP50=m.seg.map50, mask_AP50_95=m.seg.map))
    for cat in CATEGORIES:
        imgs = sorted(glob.glob(f"{cat}/test/good/*.png"))
        res = model.predict(imgs, imgsz=imgsz, conf=0.25, device=0, verbose=False, stream=True)
        confs = [float(r.boxes.conf.max()) if len(r.boxes) else 0.0 for r in res]
        fa_rows.append(dict(model=tag, category=cat, good_images=len(imgs),
                            false_alarm_rate_conf025=sum(c >= 0.25 for c in confs) / len(imgs),
                            false_alarm_rate_conf050=sum(c >= 0.50 for c in confs) / len(imgs)))
    print(tag, "done", flush=True)

per_cat = pd.DataFrame(rows); per_cat.to_csv(OUT / "per_category_ap.csv", index=False)
fa = pd.DataFrame(fa_rows); fa.to_csv(OUT / "false_alarms_on_good_parts.csv", index=False)
summary = fa.groupby("model").apply(lambda d: pd.Series({
    "good_images": d.good_images.sum(),
    "false_alarm_rate_conf025": (d.false_alarm_rate_conf025 * d.good_images).sum() / d.good_images.sum(),
    "false_alarm_rate_conf050": (d.false_alarm_rate_conf050 * d.good_images).sum() / d.good_images.sum()}), include_groups=False)
summary.to_csv(OUT / "false_alarms_summary.csv")
print(per_cat[per_cat.category == "ALL"].round(3).to_string(index=False))
print(summary.round(3).to_string())
