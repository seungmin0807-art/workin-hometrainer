# Workin Pose Prototype

CPU-friendly squat correction prototype that compares `wrong.mp4` against `correct.mp4` with `MediaPipe Pose`, samples frames at roughly 10 FPS, and generates:

- a skeleton-overlay coaching video
- frame-by-frame pose metrics and correction feedback
- a split-screen web prototype with a 3D avatar and live coaching panel

## Why MediaPipe

`MediaPipe Pose` fits this MVP better than YOLO/OpenPose because it is lighter on CPU-only laptops, provides 33 pose landmarks, and exposes approximate world-space landmarks that are usable for a single-camera 3D avatar prototype.

## Project Layout

- `correct.mp4`, `wrong.mp4`: input videos
- `scripts/analyze_videos.py`: pose extraction, alignment, scoring, and overlay rendering
- `scripts/serve.py`: local static server
- `app/`: static web UI
- `app/data/analysis.json`: generated pose and feedback data
- `app/media/wrong_overlay.mp4`: generated overlay video

## Run

From PowerShell:

```powershell
.\run.ps1
```

That command:

1. analyzes `correct.mp4` and `wrong.mp4` at about 10 FPS
2. writes processed assets into `app/data` and `app/media`
3. starts a local server at `http://127.0.0.1:8000`

## Notes

- The prototype assumes both clips show the same exercise from a similar side-view angle.
- The current feedback is tuned for a barbell back squat, because that is what the provided videos contain.
- This is a single-camera prototype, so it does not estimate true multi-view 3D reconstruction from the business plan.

