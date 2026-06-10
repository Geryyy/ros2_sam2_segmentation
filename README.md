# sam2_segmentation

Prompted image segmentation using **SAM 2.1** (Segment Anything Model 2).

The node takes **box prompts** from a `vision_msgs/Detection2DArray` (e.g. produced
by a 2D detector such as YOLO) together with a synchronized camera image, runs the
SAM 2.1 image predictor with each box as a prompt, and republishes the masks.

## Topics

| Dir | Topic | Type | Notes |
|-----|-------|------|-------|
| sub | `image_topic` (`/camera/color/image_raw`) | `sensor_msgs/Image` | RGB camera frame |
| sub | `detections_topic` (`/detections`) | `vision_msgs/Detection2DArray` | box prompts |
| pub | `~/detections` | `vision_msgs/Detection2DArray` | input boxes carried through; SAM mask confidence written to `results[0].hypothesis.score` |
| pub | `~/instance_mask` | `sensor_msgs/Image` (`mono8`/`mono16`) | instance label image, pixel `i+1` ↔ `detections[i]`, `0` = background |
| pub | `~/debug_mask` | `sensor_msgs/Image` (`mono8`) | visualization mask, foreground `255`, background `0` |

`vision_msgs/Detection2D` has no mask field in Humble, so per-object masks are
delivered via the index-aligned instance label image.

## Parameters

See [config/sam2_segmentation.yaml](config/sam2_segmentation.yaml). Key ones:

- `model_cfg` — SAM 2.1 hydra config (default `configs/sam2.1/sam2.1_hiera_b+.yaml`)
- `checkpoint` — path to the `.pt`; empty = `$SAM2_CHECKPOINT_DIR/sam2.1_hiera_base_plus.pt`
- `device` — `cuda` (falls back to CPU if unavailable)
- `sync_slop` — ApproximateTimeSynchronizer tolerance between image and boxes
- `min_mask_score` — drop masks below this SAM confidence

## Run

```bash
ros2 launch sam2_segmentation sam2_segmentation.launch.py
# or with custom params:
ros2 launch sam2_segmentation sam2_segmentation.launch.py params_file:=/path/params.yaml
```

## Experimental Video Mode

`sam2_video_segmentation_node` keeps the per-frame node unchanged and adds a
clip-based SAM2 video predictor path:

1. wait for a synced image + non-empty detection array,
2. collect `clip_length` synced frames,
3. prompt frame 0 with the YOLO boxes,
4. propagate masks through the clip,
5. publish delayed masks on `~/instance_mask` and `~/debug_mask`.

This is useful for validating SAM2 video propagation. It is not yet a fully
online tracker; output arrives after each clip is collected and processed.

```bash
ros2 launch sam2_segmentation sam2_video_segmentation.launch.py
```

## Requirements

`torch` (CUDA 12.8 / Blackwell), `torchvision`, and `sam2` are installed via pip in
the devcontainer image (see `.devcontainer/Dockerfile.vscode`), along with the
base_plus checkpoint under `/opt/sam2/checkpoints`. They are intentionally **not**
rosdep keys.
