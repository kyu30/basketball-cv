# Reproducing

Environment setup, data access, the full train/eval pipeline, and a small demo path.
Developed on Windows 11 + an NVIDIA RTX 4060 Ti (8 GB); any CUDA GPU works (lower the
`--batch` flags for less VRAM).

## 1. Environment

```bash
conda create -n basketball-cv python=3.10 -y
conda activate basketball-cv
pip install -r requirements.txt
```

`requirements.txt` pins everything, including the two GitHub-only / extra packages:
- **TrackEval** (HOTA/MOTA/IDF1) — installed from GitHub.
- **torchreid** (OSNet Re-ID) + **gdown**, **tensorboard**, **easyocr**.

PyTorch is pinned to a CUDA 12.1 build; swap the `torch`/`torchvision` lines for your
CUDA version if needed.

## 2. Data access (SportsMOT)

1. Request access to **MCG-NJU/SportsMOT** on Hugging Face and accept the license
   (CC BY-NC 4.0, non-commercial).
2. Download it into `data/`:
   ```bash
   python src/download_data.py          # -> data/sportsmot/
   python src/check_data.py             # sanity-check one clip
   ```
   Layout expected by the code: `data/sportsmot/dataset/{train,val}/.../img1/*.jpg`
   and `.../gt/gt.txt`, with `data/sportsmot/splits_txt/basketball.txt`.
3. Only the 30 clips with ground truth (official train+val) are used; the 50-clip
   test split withholds labels. All splits are rebuilt **by source video** (see
   `docs/training_procedure.md`).

`data/`, `runs/`, and large intermediates are gitignored; evaluation outputs
(`outputs/tracking/metrics.csv`, plots, tracker results, GIFs) are committed as
evidence.

## 3. Full pipeline

### A. Detector
```bash
python src/mot_to_yolo.py --stride 3                                   # MOT -> YOLO dataset
python src/train_detector.py --model yolov8n.pt --epochs 60 --name yolov8n_clipsplit
python src/train_detector.py --model yolov8s.pt --batch 8 --epochs 60 --name yolov8s_clipsplit   # larger (optional)
python src/coco_baseline.py --split test                              # naive baseline
```

### B. Tracking baselines (CI2)
```bash
python src/detect_clips.py                     # cache detections (stride 1, conf>=0.1)
python src/track_clips.py --tracker iou
python src/track_clips.py --tracker bytetrack
python src/eval_tracking.py                    # HOTA/MOTA/IDF1 -> outputs/tracking/metrics.csv
```

### C. Re-ID extension (CI3)
```bash
python src/build_reid_dataset.py               # player crops + identities (test excluded)
# fetch the Market-1501 OSNet weights the backbone is initialized from:
python -c "import gdown; gdown.download(id='1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA', output='weights/osnet_x1_0_market1501.pth')"
python src/train_reid.py --epochs 30           # -> weights/reid_basketball.pth
python src/embed_dets.py --name reid           # per-detection embeddings
python src/track_clips.py --tracker reid --name reid_motion --motion    # appearance + motion (best)
python src/eval_tracking.py
python src/analyze_switches.py --trackers iou reid          # same- vs cross-team failure analysis
python src/analyze_failure_modes.py --tracker reid_motion  # HOTA vs box size / occlusion
# ablations: off-the-shelf vs fine-tuned Re-ID, and the alpha sweep
python src/embed_dets.py --weights weights/osnet_x1_0_market1501.pth --name reid_market
python src/track_clips.py --tracker reid --name reid_market --embeds reid_market
```

### D. Ball detector + action classifier (stretch)
```bash
python src/label_segments.py                   # (manual) label action segments
python src/label_ball.py                        # (manual) label ball boxes
python src/build_ball_dataset.py
python src/train_ball_detector.py --epochs 80   # ball detector (imgsz 1280)
python src/detect_ball_clips.py                 # ball trajectory per clip
python src/track_to_trajectories.py --ball --handler
python src/train_action_model.py --features data/actions/features_ball_h.npz --bidirectional --ensemble 5
```

(The hand-labeled CSVs used in the report live under `data/actions/` and `data/ball/`;
labeling tools are resumable.)

## 4. Small demo path

**Zero-setup (no data/training needed — uses committed outputs):**
- `outputs/tracking/metrics.csv` — the full tracker comparison (HOTA/MOTA/IDF1).
- `outputs/tracking/viz/*.gif` — side-by-side tracker footage (IoU vs Re-ID+motion).
- `outputs/detector/*/summary.txt` — detector metrics.

**Minimal runnable demo (data in place + trained `best.pt`s):** produce a tracking
visualization on one clip in two commands —
```bash
python src/track_clips.py --tracker reid --name reid_motion --motion   # if not already run
python src/gif_tracks.py --clip v_00HRwkvvjtQ_c001 --start 400 --length 90 --trackers iou reid_motion
# -> outputs/tracking/viz/iou_vs_reid_motion_v_00HRwkvvjtQ_c001_f400.gif
```

## 5. Notes / gotchas

- **TrackEval + NumPy 2:** a small compatibility shim is at the top of
  `src/eval_tracking.py` (TrackEval still uses the removed `np.float`).
- **OpenCV:** `easyocr` pulls `opencv-python-headless`, which can shadow the GUI
  `opencv-python` and break the labeling tools' windows; if `cv2.imshow` fails, run
  `pip install --force-reinstall --no-deps opencv-python`.
- **8 GB VRAM:** YOLOv8s trains at `--batch 8`; the ball detector uses imgsz 1280.
  Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if you hit fragmentation.
