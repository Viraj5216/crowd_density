# gradio_yolo.py
"""
YOLOv8 Crowd Density Estimator (final)
Save as: gradio_yolo.py
Run: python gradio_yolo.py
"""

import os
import tempfile
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# ---- Config ----
MODEL_NAME = "yolov8n.pt"   # will be auto-downloaded on first run
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load YOLO model once
model = YOLO(MODEL_NAME)

# ---- Utility functions ----
def run_yolo_once(img_path, conf=0.15, imgsz=1280):
    """
    Run YOLO inference on img_path and return list of boxes.
    Each box: {'box':[x1,y1,x2,y2], 'score':float}
    """
    try:
        results = model.predict(
            source=img_path,
            conf=conf,
            imgsz=imgsz,
            device='cpu',
            classes=[0],   # person class only (COCO)
            verbose=False
        )
    except Exception as e:
        # return empty on error
        print("YOLO inference error:", e)
        return []

    if len(results) == 0:
        return []

    r = results[0]
    boxes = []
    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        xy = r.boxes.xyxy.tolist()
        confs = r.boxes.conf.tolist()
        for box, score in zip(xy, confs):
            x1, y1, x2, y2 = map(int, box)
            boxes.append({"box":[x1, y1, x2, y2], "score": float(score)})
    return boxes

def build_density_from_boxes(img_shape, boxes, grid=(32,32)):
    """
    Build a simple density map from box centers.
    Returns (heatmap_bgr, raw_density_uint8)
    """
    h, w = img_shape[:2]
    gh = int(np.ceil(h / grid[1]))
    gw = int(np.ceil(w / grid[0]))
    density = np.zeros((gh, gw), dtype=np.float32)
    for b in boxes:
        x1,y1,x2,y2 = b['box']
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        gx = min(gw - 1, cx // grid[0])
        gy = min(gh - 1, cy // grid[1])
        density[gy, gx] += 1.0
    density_up = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
    mx = density_up.max()
    den_uint8 = ((density_up / (mx + 1e-8)) * 255).astype(np.uint8) if mx > 0 else (density_up.astype(np.uint8))
    heatmap = cv2.applyColorMap(den_uint8, cv2.COLORMAP_JET)
    return heatmap, den_uint8

def annotate_image(img_bgr, boxes):
    vis = img_bgr.copy()
    for b in boxes:
        x1,y1,x2,y2 = b['box']
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, f"{b['score']:.2f}", (x1, max(12,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return vis

# ---- Main detection pipeline (used by Gradio) ----
def detect_and_visualize(image, conf=0.15, min_area_pct=0.0004):
    """
    image: PIL.Image or numpy array
    conf: YOLO confidence threshold
    min_area_pct: minimum box area as fraction of image area (filtering)
    Returns: annotated_pil, heatmap_pil, stats_text
    """
    # 1) save uploaded image to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_name = tmp.name
    tmp.close()

    # convert Gradio input to JPEG file readable by YOLO/OpenCV
    if isinstance(image, np.ndarray):
        # Gradio provides RGB numpy array
        cv2.imwrite(tmp_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        # PIL.Image
        image.save(tmp_name, format="JPEG")

    # read BGR image
    img_bgr = cv2.imread(tmp_name)
    if img_bgr is None:
        # cleanup
        try:
            os.remove(tmp_name)
        except:
            pass
        return None, None, "ERROR: failed to read uploaded image."

    h, w = img_bgr.shape[:2]
    img_area = float(h * w)

    # 2) primary YOLO pass (reasonable defaults)
    boxes = run_yolo_once(tmp_name, conf=float(conf), imgsz=1280)

    # 3) fallback pass if zero detections (more sensitive)
    tried_fallback = False
    if len(boxes) == 0:
        boxes = run_yolo_once(tmp_name, conf=max(0.08, float(conf) - 0.03), imgsz=1600)
        tried_fallback = True

    # 4) post-filter boxes by area and aspect ratio
    min_area = float(min_area_pct) * img_area
    filtered = []
    for b in boxes:
        x1,y1,x2,y2 = b['box']
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = float(bw * bh)
        aspect = bw / float(bh) if bh > 0 else 0.0
        # conditions: area threshold and reasonable aspect ratio to reduce noise
        if area >= min_area and 0.18 <= aspect <= 4.0:
            filtered.append(b)
    boxes = filtered

    # 5) produce visualization and heatmap
    vis = annotate_image(img_bgr, boxes)
    heatmap, den_raw = build_density_from_boxes(img_bgr.shape, boxes, grid=(32,32))
    overlay = cv2.addWeighted(heatmap, 0.5, img_bgr, 0.5, 0)

    # 6) save artifacts for presentation
    base = Path(tmp_name).stem
    out_annot = RESULTS_DIR / f"{base}_annotated.jpg"
    out_heat = RESULTS_DIR / f"{base}_heatmap.jpg"
    out_stats = RESULTS_DIR / f"{base}_stats.txt"
    cv2.imwrite(str(out_annot), vis)
    cv2.imwrite(str(out_heat), overlay)
    with open(out_stats, "w") as fh:
        fh.write(f"image:{tmp_name}\ncount:{len(boxes)}\n")
        fh.write("boxes:\n")
        for b in boxes:
            fh.write(f"{b['box']} score={b['score']:.3f}\n")

    # 7) prepare Gradio outputs
    annotated_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    heatmap_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    stats_lines = []
    stats_lines.append(f"Detected people (after filter): {len(boxes)}")
    if tried_fallback:
        stats_lines.append("(Fallback sensitivity pass activated)")
    stats_lines.append(f"YOLO conf threshold used: {conf}")
    stats_lines.append(f"Min box area (pct of image): {min_area_pct}")
    stats_lines.append("")
    stats_lines.append("Limitations:")
    stats_lines.append(" - Extremely dense aerial or stadium images with tiny heads may be undercounted.")
    stats_lines.append(" - For ultra-dense crowds use specialized crowd-counting networks (CSRNet/MCNN) or higher-res video.")
    stats_lines.append("")
    stats_lines.append(f"Saved outputs: {out_annot}, {out_heat}, {out_stats}")

    stats_text = "\n".join(stats_lines)

    # cleanup temp file
    try:
        os.remove(tmp_name)
    except:
        pass

    return annotated_pil, heatmap_pil, stats_text

# ---- Gradio UI ----
title = "YOLOv8 Crowd Density Estimator (final)"
desc = (
    "Upload a crowd image. YOLOv8 (person) detects people and produces an annotated output and density heatmap.\n"
    "Use the sliders to tune detection confidence and minimum box size (as fraction of image) for best results."
)

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(desc)

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload Crowd Image")
        with gr.Column():
            conf_slider = gr.Slider(label="YOLO confidence threshold", minimum=0.05, maximum=0.6, value=0.15, step=0.01)
            min_area_slider = gr.Slider(label="Min box area (fraction of image area)", minimum=0.00005, maximum=0.01, value=0.0004, step=0.00005)
            run_btn = gr.Button("Detect")
            out_annot = gr.Image(label="Annotated Output")
            out_heat = gr.Image(label="Density Heatmap")
            out_stats = gr.Textbox(label="Stats & Notes", lines=7)

    run_btn.click(fn=detect_and_visualize,
                  inputs=[inp, conf_slider, min_area_slider],
                  outputs=[out_annot, out_heat, out_stats])

# Launch server
if __name__ == "__main__":
    # If you want to share publicly set share=True, but for local demo leave False
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

