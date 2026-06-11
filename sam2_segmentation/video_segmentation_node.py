#!/usr/bin/env python3
"""Clip-based SAM 2.1 video segmentation node.

This node is intentionally separate from segmentation_node.py. It keeps the
working per-frame image predictor path intact while exposing an experimental
SAM2 video-predictor path for live ROS streams.
"""

import copy
import os
import threading
import time
from collections import OrderedDict, deque
from typing import List, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from vision_msgs.msg import Detection2DArray

import message_filters
import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2_video_predictor


def _default_checkpoint() -> str:
    ckpt_dir = os.environ.get('SAM2_CHECKPOINT_DIR', '/opt/sam2/checkpoints')
    return os.path.join(ckpt_dir, 'sam2.1_hiera_base_plus.pt')


class Sam2VideoSegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__('sam2_video_segmentation_node')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('model_cfg', 'configs/sam2.1/sam2.1_hiera_b+.yaml')
        self.declare_parameter('checkpoint', _default_checkpoint())
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('clip_length', 16)
        self.declare_parameter('clip_stride', 1)
        # Deprecated: kept so old launch overlays do not warn. Frames stay in RAM.
        self.declare_parameter('temp_root', '')
        self.declare_parameter('offload_video_to_cpu', True)
        self.declare_parameter('offload_state_to_cpu', False)

        self.image_topic = self.get_parameter('image_topic').value
        self.detections_topic = self.get_parameter('detections_topic').value
        model_cfg = self.get_parameter('model_cfg').value
        checkpoint = self.get_parameter('checkpoint').value or _default_checkpoint()
        device = self.get_parameter('device').value
        self.clip_length = max(2, int(self.get_parameter('clip_length').value))
        self.clip_stride = max(1, int(self.get_parameter('clip_stride').value))
        self.offload_video_to_cpu = bool(self.get_parameter('offload_video_to_cpu').value)
        self.offload_state_to_cpu = bool(self.get_parameter('offload_state_to_cpu').value)

        if device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but not available; using CPU.')
            device = 'cpu'
        self.device = torch.device(device)
        self._autocast_dtype = (
            torch.bfloat16 if self.device.type == 'cuda' else torch.float32)

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f'SAM 2 checkpoint not found: {checkpoint}. '
                'Set the "checkpoint" parameter or SAM2_CHECKPOINT_DIR.')

        self.get_logger().info(
            f'Loading SAM 2.1 video predictor '
            f'(cfg={model_cfg}, ckpt={checkpoint}) on {device} ...')
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=self.device)
        self.get_logger().info('SAM 2.1 video predictor ready.')

        self.bridge = CvBridge()
        self.det_pub = self.create_publisher(Detection2DArray, '~/detections', 10)
        self.mask_pub = self.create_publisher(Image, '~/instance_mask', 10)
        self.debug_mask_pub = self.create_publisher(Image, '~/debug_mask', 10)
        self.timing_pub = self.create_publisher(Float64MultiArray, '~/timing', 10)
        self.timing_frames_pub = self.create_publisher(Float64, '~/timing/frames', 10)
        self.timing_objects_pub = self.create_publisher(Float64, '~/timing/objects', 10)
        self.timing_total_ms_pub = self.create_publisher(Float64, '~/timing/total_ms', 10)
        self.timing_per_frame_ms_pub = self.create_publisher(
            Float64, '~/timing/per_frame_ms', 10)

        self._lock = threading.Lock()
        self._processing = False
        self._buffer = deque(maxlen=self.clip_length)
        self._sample_seq = 0
        self._last_started_seq = -1

        queue_size = int(self.get_parameter('sync_queue_size').value)
        slop = float(self.get_parameter('sync_slop').value)
        self.image_sub = message_filters.Subscriber(
            self, Image, self.image_topic, qos_profile=qos_profile_sensor_data)
        self.det_sub = message_filters.Subscriber(
            self, Detection2DArray, self.detections_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.det_sub], queue_size=queue_size, slop=slop)
        self.sync.registerCallback(self.on_synced)

        self.get_logger().info(
            f'Subscribed to image="{self.image_topic}", boxes="{self.detections_topic}", '
            f'clip_length={self.clip_length}, clip_stride={self.clip_stride}.')

    @staticmethod
    def _boxes_from_detections(det_msg: Detection2DArray) -> np.ndarray:
        boxes = []
        for det in det_msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            boxes.append([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    def on_synced(self, image_msg: Image, det_msg: Detection2DArray) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        sample = (rgb.copy(), copy.deepcopy(image_msg), copy.deepcopy(det_msg))
        self._maybe_start_latest_window(sample)

    def _maybe_start_latest_window(self, sample=None) -> None:
        with self._lock:
            if sample is not None:
                self._sample_seq += 1
                self._buffer.append((self._sample_seq, *sample))

            if self._processing or len(self._buffer) < self.clip_length:
                return

            clip = list(self._buffer)
            prompt_det_msg = clip[0][3]
            if not prompt_det_msg.detections:
                return

            start_seq = clip[0][0]
            advanced = start_seq - self._last_started_seq
            if advanced <= 0 or advanced < self.clip_stride:
                return
            self._last_started_seq = start_seq
            self._processing = True

        self.get_logger().info(
            f'Started SAM2 video window from {len(prompt_det_msg.detections)} box prompts.')
        threading.Thread(
            target=self._process_clip,
            args=(clip, copy.deepcopy(prompt_det_msg)),
            daemon=True).start()

    def _process_clip(
        self,
        clip: List[Tuple[np.ndarray, Image, Detection2DArray]],
        prompt_det_msg: Detection2DArray,
    ) -> None:
        t_start = time.perf_counter()
        try:
            boxes = self._boxes_from_detections(prompt_det_msg)
            if len(boxes) == 0:
                return

            self.get_logger().info(
                f'Processing SAM2 video clip frames={len(clip)} objects={len(boxes)} ...')
            with torch.inference_mode(), torch.autocast(
                    self.device.type, dtype=self._autocast_dtype):
                inference_state = self._init_state_from_clip(clip)
                for obj_idx, box in enumerate(boxes, start=1):
                    self.predictor.add_new_points_or_box(
                        inference_state,
                        frame_idx=0,
                        obj_id=obj_idx,
                        box=box)
                for frame_idx, obj_ids, video_res_masks in self.predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=0,
                        max_frame_num_to_track=len(clip)):
                    self._publish_frame_result(
                        clip[frame_idx][2],
                        prompt_det_msg,
                        obj_ids,
                        video_res_masks)

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            timing_msg = Float64MultiArray()
            timing_msg.data = [
                float(len(clip)),
                float(len(boxes)),
                elapsed_ms,
                elapsed_ms / float(len(clip)),
            ]
            self.timing_pub.publish(timing_msg)
            self.timing_frames_pub.publish(Float64(data=float(len(clip))))
            self.timing_objects_pub.publish(Float64(data=float(len(boxes))))
            self.timing_total_ms_pub.publish(Float64(data=elapsed_ms))
            self.timing_per_frame_ms_pub.publish(
                Float64(data=elapsed_ms / float(len(clip))))
            self.get_logger().info(f'Finished SAM2 video clip in {elapsed_ms:.1f} ms.')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'SAM2 video clip failed: {exc}')
        finally:
            with self._lock:
                self._processing = False
            self._maybe_start_latest_window()

    def _init_state_from_clip(
        self,
        clip: List[Tuple[np.ndarray, Image, Detection2DArray]],
    ):
        video_height, video_width = clip[0][1].shape[:2]
        image_size = int(self.predictor.image_size)

        frames = np.stack([rgb for _, rgb, _, _ in clip], axis=0)
        images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False)

        storage_device = torch.device('cpu') if self.offload_video_to_cpu else self.device
        images = images.to(storage_device)
        img_mean = torch.tensor(
            (0.485, 0.456, 0.406),
            dtype=torch.float32,
            device=storage_device)[:, None, None]
        img_std = torch.tensor(
            (0.229, 0.224, 0.225),
            dtype=torch.float32,
            device=storage_device)[:, None, None]
        images = (images - img_mean) / img_std

        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(clip)
        inference_state["offload_video_to_cpu"] = self.offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = self.offload_state_to_cpu
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = self.device
        inference_state["storage_device"] = (
            torch.device("cpu") if self.offload_state_to_cpu else self.device)
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}

        self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def _publish_frame_result(
        self,
        image_msg: Image,
        prompt_det_msg: Detection2DArray,
        obj_ids,
        video_res_masks,
    ) -> None:
        height = image_msg.height
        width = image_msg.width
        label_dtype = np.uint16 if len(obj_ids) >= 255 else np.uint8
        instance = np.zeros((height, width), dtype=label_dtype)

        masks = video_res_masks.detach().cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0, :, :]

        out = Detection2DArray()
        out.header = image_msg.header
        for label_value, obj_id in enumerate(obj_ids, start=1):
            mask = masks[label_value - 1] > 0.0
            instance[mask] = label_value
            src_index = int(obj_id) - 1
            if 0 <= src_index < len(prompt_det_msg.detections):
                out.detections.append(copy.deepcopy(prompt_det_msg.detections[src_index]))
        for det in out.detections:
            det.header = image_msg.header

        encoding = 'mono16' if label_dtype == np.uint16 else 'mono8'
        mask_msg = self.bridge.cv2_to_imgmsg(instance, encoding=encoding)
        mask_msg.header = image_msg.header
        debug_mask_msg = self.bridge.cv2_to_imgmsg(
            ((instance > 0) * 255).astype(np.uint8), encoding='mono8')
        debug_mask_msg.header = image_msg.header

        self.det_pub.publish(out)
        self.mask_pub.publish(mask_msg)
        self.debug_mask_pub.publish(debug_mask_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Sam2VideoSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
