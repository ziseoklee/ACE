import os

os.environ["MPLBACKEND"] = "Agg"  # Set matplotlib to non-GUI backend

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import groundingdino
import numpy as np
import torch
import torchvision
from groundingdino.util.inference import Model
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, SamModel, SamProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)-12s - %(levelname)-8s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Class to store evaluation metrics and statistics."""

    # CLIP metrics
    clip_record: float = 0.0
    clip_count: int = 0
    local_clip_record: float = 0.0
    local_clip_count: int = 0

    # mIoU metrics
    miou_record: float = 0.0
    miou_count: int = 0
    miou_level_record: list[float] = field(default_factory=lambda: [0.0] * 5)
    miou_level_count: list[int] = field(default_factory=lambda: [0] * 5)

    # Success rate metrics
    success_record: float = 0.0
    success_count: int = 0
    success_level_record: list[int] = field(default_factory=lambda: [0] * 5)
    success_level_count: list[int] = field(default_factory=lambda: [0] * 5)

    # Instance success rate metrics
    inst_success_count: int = 0
    inst_count: int = 0
    inst_success_level_count: list[int] = field(default_factory=lambda: [0] * 5)
    inst_level_count: list[int] = field(default_factory=lambda: [0] * 5)


@dataclass
class ColorRange:
    """Class to define HSV color ranges."""

    lower: np.ndarray
    upper: np.ndarray


class ColorDetector:
    """Handles color detection in HSV color space."""

    COLOR_RANGES = {
        "red": [
            ColorRange(np.array([0, 50, 70]), np.array([9, 255, 255])),
            ColorRange(np.array([159, 50, 70]), np.array([180, 255, 255])),
        ],
        "blue": ColorRange(np.array([90, 50, 70]), np.array([128, 255, 255])),
        "yellow": ColorRange(np.array([25, 50, 70]), np.array([35, 255, 255])),
        "green": ColorRange(np.array([36, 50, 70]), np.array([89, 255, 255])),
        "black": ColorRange(np.array([0, 0, 0]), np.array([180, 255, 30])),
        "white": ColorRange(np.array([0, 0, 221]), np.array([180, 43, 255])),
        "brown": ColorRange(np.array([6, 43, 35]), np.array([25, 255, 255])),
    }

    @classmethod
    def check_color(cls, image: np.ndarray, color: str) -> np.ndarray:
        """Check if pixels in image match the specified color."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_ranges = cls.COLOR_RANGES.get(color)

        if not color_ranges:
            raise ValueError(f"Unsupported color: {color}")

        if isinstance(color_ranges, list):
            mask = np.zeros([512, 512], np.uint8)
            for color_range in color_ranges:
                result_mask = cv2.inRange(hsv_image, color_range.lower, color_range.upper) / 255
                mask = np.logical_or(result_mask, mask).astype(np.int_)
            mask = mask * 255
        else:
            mask = cv2.inRange(hsv_image, color_ranges.lower, color_ranges.upper)

        return mask


# ImageNet templates for CLIP evaluation
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


class MIGBenchEvaluator:
    """Main class for MIG Bench evaluation"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = EvaluationMetrics()

        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize all required models using original GroundingDINO and transformers SAM."""
        logger.info("Initializing models...")

        # GroundingDINO
        grounding_dino_config = str(Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py")
        grounding_dino_checkpoint = "./data/groundingdino_swint_ogc.pth"
        self.grounding_dino_model = Model(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint,
        )
        logger.info("✓ GroundingDino model loaded successfully")

        # SAM
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")
        # Move to device
        self.sam_model = torch.nn.Module.to(self.sam_model, self.device)
        self.sam_model.eval()
        logger.info("✓ SAM model loaded successfully")

        # CLIP (only if needed)
        if self.args.need_clip_score or self.args.need_local_clip:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            # Move to device
            self.clip_model = torch.nn.Module.to(self.clip_model, self.device)
            self.clip_model.eval()
            logger.info("✓ CLIP model loaded successfully")
        else:
            self.clip_model = None
            self.clip_processor = None
            logger.info("CLIP model not loaded (not needed)")

        logger.info("All models initialized successfully.")

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load the JSONL dataset."""
        dataset = []
        with open(self.args.bench_file_path, "r") as f:
            for line in f:
                dataset.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(dataset)} samples from {self.args.bench_file_path}")
        return dataset

    def find_image_path(self, idx: int, itr: int, level: int, prompt: str, image_list: list[str]) -> list[str] | None:
        """Find the image path for given parameters."""
        image_name_pattern = rf"^{idx}_{itr}_{level}_{re.escape(prompt)}.*\.(?:png|jpg)$"
        matching_paths = [path for path in image_list if re.match(image_name_pattern, Path(path).name)]
        # drop filename ends with _layout.(png|jpg)
        matching_paths = [path for path in matching_paths if not re.search(r"_layout\.(?:png|jpg)$", path)]
        logger.debug(f"Found {len(matching_paths)} matching images for prompt '{prompt}' at iteration {itr}")

        if len(matching_paths) == 0:
            logger.warning(f"Image for prompt '{prompt}' at iteration {itr} not found")
            return None
        # elif len(matching_paths) > 1:
        #     logger.warning(f"Multiple images found for prompt '{prompt}' at iteration {itr}, using the first one")

        return [str(Path(self.args.image_dir) / matching_paths[i]) for i in range(len(matching_paths))]

    def calculate_clip_score(self, image: np.ndarray, prompt: str, use_templates: bool = False) -> float:
        """Calculate CLIP score between image and prompt."""
        if self.clip_model is None or self.clip_processor is None:
            return 0.0

        prompt_list = []
        if use_templates:
            for template in IMAGENET_TEMPLATES:
                prompt_list.append(template.format(prompt))
        else:
            prompt_list.append(prompt)

        inputs = self.clip_processor(text=prompt_list, images=image, return_tensors="pt", padding=True)

        # Move tensors to device
        for key, value in inputs.items():
            if hasattr(value, "to"):
                inputs[key] = value.to(self.device)

        with torch.inference_mode():
            outputs = self.clip_model(**inputs)

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return torch.mean(outputs.logits_per_image).cpu().item()

    def detect_objects_with_grounding_dino(self, image: np.ndarray, text_prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Detect objects using original GroundingDINO implementation."""
        # Detect objects using GroundingDINO
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=[text_prompt],
            box_threshold=0.25,
            text_threshold=0.25,
        )

        # Apply NMS
        if detections.xyxy.shape[0] > 0:
            nms_idx = (
                torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    0.8,
                )
                .numpy()
                .tolist()
            )

            boxes = detections.xyxy[nms_idx]
            scores = (
                detections.confidence[nms_idx]
                if hasattr(detections, "confidence") and detections.confidence is not None
                else np.ones(len(nms_idx))
            )
        else:
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])

        return boxes, scores

    def segment_object_with_sam(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Segment object using SAM given a bounding box."""
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare input boxes - SAM expects boxes in [x_min, y_min, x_max, y_max] format
        input_boxes = [[bbox.tolist()]]  # Batch of boxes

        inputs = self.sam_processor(image_rgb, input_boxes=input_boxes, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.sam_model(**inputs)

        # Post-process masks
        masks = self.sam_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]

        # Select the mask with highest IoU score
        iou_scores = outputs.iou_scores.cpu().numpy()[0]
        best_mask_idx = np.argmax(iou_scores)
        best_mask = masks[0, best_mask_idx].numpy().astype(np.uint8)

        return best_mask

    def calculate_iou(self, pred_bbox: np.ndarray, gt_bbox: np.ndarray) -> float:
        """Calculate IoU between predicted and ground truth bounding boxes."""
        # Ensure boxes are in [x1, y1, x2, y2] format
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox

        # Calculate intersection
        x1 = max(pred_x1, gt_x1)
        y1 = max(pred_y1, gt_y1)
        x2 = min(pred_x2, gt_x2)
        y2 = min(pred_y2, gt_y2)

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union_area = pred_area + gt_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def evaluate_object_detection(
        self, image: np.ndarray, prompt: str, gt_bbox: np.ndarray, color: str, miou_threshold: float = 0.5
    ) -> tuple[bool, bool, float]:
        """
        Evaluate object detection and attribute verification for a single object.

        Returns:
            tuple: (success_flag, attr_flag, miou)
        """
        # Detect objects using original GroundingDINO (same as eval_mig_refactored.py)
        boxes, scores = self.detect_objects_with_grounding_dino(image, prompt)

        # Check if any objects were detected
        if len(boxes) == 0:
            return False, False, 0.0

        # Calculate IoU with ground truth
        best_iou = 0.0
        for pred_bbox in boxes:
            iou = self.calculate_iou(pred_bbox, gt_bbox)
            best_iou = max(best_iou, iou)

        # If IoU is below threshold, return failure
        if best_iou < miou_threshold:
            return False, False, best_iou

        # Object detection successful, now check attributes
        success_flag = True

        # Segment the object using ground truth bbox (for more accurate attribute checking)
        object_mask = self.segment_object_with_sam(image, gt_bbox)

        # Apply mask to isolate object
        masked_image = self._apply_mask_to_image(image, object_mask)

        # Check color attribute
        color_match = self._check_color_attribute(masked_image, object_mask, color)
        attr_flag = True if color_match else False

        # Replicate the original script's behavior: reset IoU if attribute check fails
        if not color_match:
            logger.debug(f"Color attribute check failed for {prompt} (expected: {color})")
            logger.debug(f"Resetting IoU to zero for {prompt}")
            best_iou = 0.0
        else:
            logger.debug(f"Color attribute check passed for {prompt}")

        return success_flag, attr_flag, best_iou

    def _apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image, setting background to gray."""
        if mask.ndim == 3:
            mask = mask[:, :, 0]  # Take first channel if 3D

        masked_image = image * mask[:, :, np.newaxis]
        background = (1 - mask)[:, :, np.newaxis] * 127  # Gray background
        return (masked_image + background).astype(np.uint8)

    def _check_color_attribute(self, image: np.ndarray, mask: np.ndarray, color: str) -> bool:
        """Check if the masked region matches the expected color."""
        try:
            color_mask = ColorDetector.check_color(image, color)
            color_mask_normalized = color_mask / 255.0

            if mask.ndim == 3:
                mask = mask[:, :, 0]  # Take first channel if 3D

            # Calculate overlap between object mask and color mask
            overlap = np.logical_and(mask, color_mask_normalized).sum()
            total_object_pixels = mask.sum()

            if total_object_pixels == 0:
                return False

            # Require at least 20% of object pixels to match the expected color
            color_ratio = overlap / total_object_pixels
            return color_ratio >= 0.2
        except ValueError as e:
            logger.warning(f"Color check failed: {e}")
            return False

    def evaluate_single_image(self, sample: dict[str, Any], image_path: str) -> None:
        """Evaluate a single image against its ground truth data."""
        # Load and resize image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        if image.shape[0] != 512:
            image = cv2.resize(image, (512, 512))

        prompt = sample["prompt"]
        phrases = sample["phrases"]
        # Convert the data structure: combine bounding_boxes with pre-parsed obj/color data
        bboxes = sample["bounding_boxes"]
        level = sample["level"]

        # Build gt_bbox_list using pre-parsed object and color information
        gt_bbox_list = []
        num_objects = sample["num_objects"]

        for i in range(num_objects):
            obj_key = f"expected_obj{i + 1}"
            color_key = f"color{i + 1}"

            # Get pre-parsed object and color
            obj_label = sample.get(obj_key, "unknown")
            attr = sample.get(color_key, "unknown")

            gt_bbox_list.append(
                {
                    "label": phrases[i],
                    "bbox": bboxes[i],
                    "obj_label": obj_label,
                    "attr": attr,
                }
            )

        # Calculate global CLIP score
        if self.args.need_clip_score:
            clip_score = self.calculate_clip_score(image, prompt)
            self.metrics.clip_record += clip_score
            self.metrics.clip_count += 1

        # Evaluate each object in the image
        if self._needs_instance_evaluation:
            image_success_obj = True
            image_success_attr = True

            for gt_instance in gt_bbox_list:
                # Use pre-parsed object and attribute data
                obj_label = gt_instance["obj_label"]
                attr = gt_instance["attr"]
                label_with_attr = gt_instance["label"]  # For display/logging purposes

                gt_bbox = np.array(gt_instance["bbox"]) * 512

                # Evaluate object detection and attributes
                success_obj, success_attr, miou = self.evaluate_object_detection(
                    image, obj_label, gt_bbox, attr, self.args.miou_threshold
                )

                image_success_obj = image_success_obj and success_obj
                image_success_attr = image_success_attr and success_attr

                # Update metrics
                self._update_instance_metrics(success_obj, success_attr, miou, level)

                # Calculate local CLIP score
                if self.args.need_local_clip:
                    self._evaluate_local_clip(image, gt_instance, label_with_attr)

            # Update image-level success metrics
            if self.args.need_success_ratio:
                self._update_image_success_metrics(image_success_obj and image_success_attr, level)

    @property
    def _needs_instance_evaluation(self) -> bool:
        """Check if instance-level evaluation is needed."""
        return (
            self.args.need_success_ratio
            or self.args.need_local_clip
            or self.args.need_instance_success_ratio
            or self.args.need_miou_score
        )

    def _update_instance_metrics(self, success_obj: bool, success_attr: bool, miou: float, level: int):
        """Update instance-level metrics."""
        if self.args.need_miou_score:
            self.metrics.miou_record += miou
            self.metrics.miou_count += 1
            self.metrics.miou_level_record[level] += miou
            self.metrics.miou_level_count[level] += 1

            # Log intermediate miou_level values
            current_miou_level = [
                self.metrics.miou_level_record[i] / max(self.metrics.miou_level_count[i], 1) for i in range(5)
            ]
            logger.debug(
                f"Intermediate mIoU level {level}: {miou:.4f}, Running avg per level: {[f'{x:.4f}' for x in current_miou_level]}"
            )

        if self.args.need_instance_success_ratio:
            self.metrics.inst_count += 1
            self.metrics.inst_level_count[level] += 1
            if success_obj and success_attr:
                self.metrics.inst_success_count += 1
                self.metrics.inst_success_level_count[level] += 1

            # Log intermediate inst_level_success_ratio values
            current_inst_level_success_ratio = [
                self.metrics.inst_success_level_count[i] / max(self.metrics.inst_level_count[i], 1) for i in range(5)
            ]
            logger.debug(
                f"Intermediate inst_level_success_ratio level {level}: success={success_obj and success_attr}, Running avg per level: {[f'{x:.4f}' for x in current_inst_level_success_ratio]}"
            )

    def _update_image_success_metrics(self, success: bool, level: int):
        """Update image-level success metrics."""
        self.metrics.success_count += 1
        self.metrics.success_level_count[level] += 1
        if success:
            self.metrics.success_record += 1
            self.metrics.success_level_record[level] += 1

        # Log intermediate success_level_ratio values
        current_success_level_ratio = [
            self.metrics.success_level_record[i] / max(self.metrics.success_level_count[i], 1) for i in range(5)
        ]
        logger.debug(
            f"Image success level {level}: success={success}, Running avg per level: {[f'{x:.4f}' for x in current_success_level_ratio]}"
        )

    def _evaluate_local_clip(self, image: np.ndarray, gt_instance: dict[str, Any], label: str):
        """Evaluate local CLIP score for a cropped object."""
        bbox = gt_instance["bbox"]
        y1, y2 = int(512 * bbox[1]), int(512 * bbox[3])
        x1, x2 = int(512 * bbox[0]), int(512 * bbox[2])

        if y2 > y1 and x2 > x1:  # Valid bbox
            cropped_image = image[y1:y2, x1:x2]
            if cropped_image.size > 0:
                cropped_image = cv2.resize(cropped_image, (512, 512))
                local_clip_score = self.calculate_clip_score(cropped_image, label, use_templates=True)
                self.metrics.local_clip_record += local_clip_score
                self.metrics.local_clip_count += 1

                logger.debug(
                    f"Local CLIP score for '{label}': {local_clip_score:.4f}, Running avg: {self.metrics.local_clip_record / self.metrics.local_clip_count:.4f}"
                )
            else:
                logger.debug(f"Empty cropped image for '{label}', skipping local CLIP evaluation")
        else:
            logger.debug(f"Invalid bbox for '{label}': [{x1}, {y1}, {x2}, {y2}], skipping local CLIP evaluation")

    def run_evaluation(self) -> dict[str, Any]:
        """Run the complete evaluation pipeline."""
        logger.info("Starting MIG Bench evaluation...")
        logger.info(f"Using device: {self.device}")
        logger.info(
            f"Evaluation settings - miou_threshold: {self.args.miou_threshold}, num_iters: {self.args.num_iters}"
        )

        # Load dataset
        dataset = self.load_dataset()

        # Get list of image files
        if not os.path.exists(self.args.image_dir) or len(os.listdir(self.args.image_dir)) == 0:
            logger.error("No generated images found.")
            return {}

        image_files = os.listdir(self.args.image_dir)

        # Process each sample
        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating images")):
            if idx % 20 == 0:  # Log every 20th sample to track progress
                logger.debug(
                    f"Processing sample {idx}/{len(dataset)}: level={sample['level']}, prompt='{sample['prompt'][:50]}...'"
                )
            for itr in range(self.args.num_iters):
                image_path_list = self.find_image_path(idx, itr, sample["level"], sample["prompt"], image_files)
                if image_path_list:
                    for image_path in image_path_list:
                        self.evaluate_single_image(sample, image_path)

        # Generate final results
        return self.generate_results()

    def generate_results(self) -> dict[str, Any]:
        """Generate final evaluation results."""
        results = {
            "metric_name": self.args.metric_name,
            "image_path": self.args.image_dir,
        }

        # CLIP scores
        if self.args.need_clip_score and self.metrics.clip_count > 0:
            results["clip_score"] = self.metrics.clip_record / self.metrics.clip_count

        if self.args.need_local_clip and self.metrics.local_clip_count > 0:
            results["local_clip_score"] = self.metrics.local_clip_record / self.metrics.local_clip_count

        # Success ratios
        if self.args.need_success_ratio and self.metrics.success_count > 0:
            results["success_ratio"] = self.metrics.success_record / self.metrics.success_count
            results["success_level_ratio"] = [
                self.metrics.success_level_record[i] / max(self.metrics.success_level_count[i], 1) for i in range(5)
            ]

        # Instance success ratios
        if self.args.need_instance_success_ratio and self.metrics.inst_count > 0:
            results["inst_success_ratio"] = self.metrics.inst_success_count / self.metrics.inst_count
            results["inst_level_success_ratio"] = [
                self.metrics.inst_success_level_count[i] / max(self.metrics.inst_level_count[i], 1) for i in range(5)
            ]

        # mIoU scores
        if self.args.need_miou_score and self.metrics.miou_count > 0:
            results["miou"] = self.metrics.miou_record / self.metrics.miou_count
            results["miou_level"] = [
                self.metrics.miou_level_record[i] / max(self.metrics.miou_level_count[i], 1) for i in range(5)
            ]

        return results


def main():
    """Main function to run MIG Bench evaluation"""
    parser = argparse.ArgumentParser(description="MIG Bench Evaluation using Original GroundingDINO + Transformers SAM")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to generated image files")
    parser.add_argument("--metric_name", type=str, required=True, help="Name for the metric results")
    parser.add_argument("--need_clip_score", action="store_true", help="Calculate CLIP score")
    parser.add_argument("--need_local_clip", action="store_true", help="Calculate local CLIP score")
    parser.add_argument("--need_success_ratio", action="store_true", help="Calculate success ratio")
    parser.add_argument("--need_instance_success_ratio", action="store_true", help="Calculate instance success ratio")
    parser.add_argument("--need_miou_score", action="store_true", help="Calculate mIoU score")
    parser.add_argument("--miou_threshold", type=float, default=0.5, help="IoU threshold for success")
    parser.add_argument(
        "--bench_file_path",
        type=str,
        default="./data/mig_bench.jsonl",
        help="Path to benchmark JSONL file",
    )
    parser.add_argument("--num_iters", type=int, default=8, help="Number of iterations per prompt")
    parser.add_argument(
        "--log_level", choices=["debug", "info", "warning", "error"], default="info", help="Logging level"
    )
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output results")

    args = parser.parse_args()

    # Update log level
    logger.setLevel(getattr(logging, args.log_level.upper()))
    logger.info(f"Log level set to {args.log_level.upper()}")

    # Create evaluator and run evaluation
    evaluator = MIGBenchEvaluator(args)
    results = evaluator.run_evaluation()

    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, list):
            formatted_values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in value]
            print(f"{key}: {formatted_values}")
        else:
            print(f"{key}: {value}")

    # Save results to file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/metric_{args.metric_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
