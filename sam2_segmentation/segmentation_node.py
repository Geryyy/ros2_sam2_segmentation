#!/usr/bin/env python3
"""Prompted SAM 2.1 segmentation node.

Subscribes to a camera image and a vision_msgs/Detection2DArray of box prompts
(e.g. produced by a 2D object detector), runs the SAM 2.1 image predictor with
each detection's bounding box as a box prompt, and republishes:

  * <ns>/detections    vision_msgs/Detection2DArray  - input boxes carried
                                                       through, with the SAM mask
                                                       confidence stored as the
                                                       hypothesis score.
  * <ns>/instance_mask sensor_msgs/Image (mono8/mono16) - instance label image
                                                       where pixel value (i + 1)
                                                       corresponds to detections[i].
                                                       0 is background.

vision_msgs/Detection2D has no mask field in Humble, so the per-object masks are
delivered via the companion instance label image, index-aligned with the
detections array.
"""

import os

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

import message_filters

# Torch / SAM 2 are heavy imports; keep them at module load so failures surface
# clearly at node start rather than on the first callback.
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _default_checkpoint() -> str:
    ckpt_dir = os.environ.get('SAM2_CHECKPOINT_DIR', '/opt/sam2/checkpoints')
    return os.path.join(ckpt_dir, 'sam2.1_hiera_base_plus.pt')


class Sam2SegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__('sam2_segmentation_node')

        # --- Parameters ---------------------------------------------------
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('model_cfg', 'configs/sam2.1/sam2.1_hiera_b+.yaml')
        self.declare_parameter('checkpoint', _default_checkpoint())
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('multimask_output', False)
        # ApproximateTimeSynchronizer slop (seconds) between image and boxes.
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('sync_queue_size', 10)
        # Score below which a SAM mask is dropped.
        self.declare_parameter('min_mask_score', 0.0)

        self.image_topic = self.get_parameter('image_topic').value
        self.detections_topic = self.get_parameter('detections_topic').value
        model_cfg = self.get_parameter('model_cfg').value
        checkpoint = self.get_parameter('checkpoint').value or _default_checkpoint()
        device = self.get_parameter('device').value
        self.multimask_output = bool(self.get_parameter('multimask_output').value)
        self.min_mask_score = float(self.get_parameter('min_mask_score').value)

        if device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but not available; using CPU.')
            device = 'cpu'
        self.device = torch.device(device)

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f'SAM 2 checkpoint not found: {checkpoint}. '
                'Set the "checkpoint" parameter or SAM2_CHECKPOINT_DIR.')

        self.get_logger().info(
            f'Loading SAM 2.1 (cfg={model_cfg}, ckpt={checkpoint}) on {device} ...')
        sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self._autocast_dtype = (
            torch.bfloat16 if self.device.type == 'cuda' else torch.float32)
        self.get_logger().info('SAM 2.1 model ready.')

        self.bridge = CvBridge()

        # --- Publishers ---------------------------------------------------
        self.det_pub = self.create_publisher(Detection2DArray, '~/detections', 10)
        self.mask_pub = self.create_publisher(Image, '~/instance_mask', 10)

        # --- Synchronized subscribers ------------------------------------
        queue_size = int(self.get_parameter('sync_queue_size').value)
        slop = float(self.get_parameter('sync_slop').value)
        self.image_sub = message_filters.Subscriber(self, Image, self.image_topic)
        self.det_sub = message_filters.Subscriber(
            self, Detection2DArray, self.detections_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.det_sub], queue_size=queue_size, slop=slop)
        self.sync.registerCallback(self.on_synced)

        self.get_logger().info(
            f'Subscribed to image="{self.image_topic}", '
            f'boxes="{self.detections_topic}" (slop={slop}s).')

    # --- Helpers ----------------------------------------------------------
    @staticmethod
    def _boxes_from_detections(detections) -> np.ndarray:
        """Return an (N, 4) array of XYXY boxes from a Detection2DArray."""
        boxes = []
        for det in detections.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            boxes.append([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    def on_synced(self, image_msg: Image, det_msg: Detection2DArray) -> None:
        if not det_msg.detections:
            # Nothing to prompt with; still publish an empty mask for consumers.
            self._publish_empty(image_msg, det_msg)
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        except Exception as exc:  # noqa: BLE001 - report and skip frame
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        boxes = self._boxes_from_detections(det_msg)
        height, width = rgb.shape[:2]

        try:
            with torch.inference_mode(), torch.autocast(
                    self.device.type, dtype=self._autocast_dtype):
                self.predictor.set_image(rgb)
                masks, scores, _ = self.predictor.predict(
                    box=boxes, multimask_output=self.multimask_output)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'SAM 2 inference failed: {exc}')
            return

        # Normalize mask/score shapes across single- vs multi-box outputs.
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(len(boxes), -1)
        if masks.ndim == 3:  # single box -> (C, H, W); add batch dim
            masks = masks[None, ...]
        # masks: (N, C, H, W). Pick the best channel per box.
        best = np.argmax(scores, axis=1)
        n = masks.shape[0]

        label_dtype = np.uint16 if n >= 255 else np.uint8
        instance = np.zeros((height, width), dtype=label_dtype)

        out = Detection2DArray()
        out.header = image_msg.header
        for i in range(n):
            score = float(scores[i, best[i]])
            if score < self.min_mask_score:
                continue
            mask = masks[i, best[i]] > 0.0
            instance[mask] = len(out.detections) + 1
            det = det_msg.detections[i]
            if det.results:
                det.results[0].hypothesis.score = score
            out.detections.append(det)

        encoding = 'mono16' if label_dtype == np.uint16 else 'mono8'
        mask_msg = self.bridge.cv2_to_imgmsg(instance, encoding=encoding)
        mask_msg.header = image_msg.header

        self.det_pub.publish(out)
        self.mask_pub.publish(mask_msg)

    def _publish_empty(self, image_msg: Image, det_msg: Detection2DArray) -> None:
        try:
            height = image_msg.height
            width = image_msg.width
            instance = np.zeros((height, width), dtype=np.uint8)
            mask_msg = self.bridge.cv2_to_imgmsg(instance, encoding='mono8')
            mask_msg.header = image_msg.header
            out = Detection2DArray()
            out.header = image_msg.header
            self.det_pub.publish(out)
            self.mask_pub.publish(mask_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to publish empty result: {exc}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Sam2SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
