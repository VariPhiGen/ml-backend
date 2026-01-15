import os
import logging

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from control_models.base import ControlModel
from control_models.choices import ChoicesModel
from control_models.rectangle_labels import RectangleLabelsModel
from control_models.rectangle_labels_obb import RectangleLabelsObbModel
from control_models.polygon_labels import PolygonLabelsModel
from control_models.keypoint_labels import KeypointLabelsModel
from control_models.video_rectangle import VideoRectangleModel
from control_models.timeline_labels import TimelineLabelsModel
from typing import List, Dict, Optional

# Import additional dependencies for hybrid mode
from uuid import uuid4
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path


logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

# Test imports for hybrid mode
try:
    # These imports are only needed when hybrid mode is enabled
    import pathlib
    import torch
    from groundingdino.util.inference import load_model, load_image, predict
    from groundingdino.util import box_ops
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False

# Register available model classes
available_model_classes = [
    ChoicesModel,
    RectangleLabelsModel,
    RectangleLabelsObbModel,
    PolygonLabelsModel,
    KeypointLabelsModel,
    VideoRectangleModel,
    TimelineLabelsModel,
]


class YOLO(LabelStudioMLBase):
    """Label Studio ML Backend based on Ultralytics YOLO"""

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "yolo")

        # Initialize Grounding DINO if available
        if GROUNDING_DINO_AVAILABLE:
            self._init_grounding_dino()
        else:
            self.grounding_dino_model = None

    def _init_grounding_dino(self):
        """Initialize Grounding DINO model"""
        try:
            GROUNDING_DINO_CONFIG = os.getenv('GROUNDING_DINO_CONFIG', 'GroundingDINO_SwinT_OGC.py')
            GROUNDING_DINO_WEIGHTS = os.getenv('GROUNDING_DINO_WEIGHTS', 'groundingdino_swint_ogc.pth')

            self.grounding_dino_model = load_model(
                pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', "./GroundingDINO")) / "groundingdino" / "config" / GROUNDING_DINO_CONFIG,
                pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', "./GroundingDINO")) / "weights" / GROUNDING_DINO_WEIGHTS
            )

            self.grounding_dino_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.grounding_dino_box_threshold = float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.3"))
            self.grounding_dino_text_threshold = float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.25"))

            logger.info(f"Grounding DINO initialized on device: {self.grounding_dino_device}")
        except Exception as e:
            logger.warning(f"Failed to initialize Grounding DINO: {e}")
            self.grounding_dino_model = None

    def detect_control_models(self) -> List[ControlModel]:
        """Detect control models based on the labeling config.
        Control models are used to predict regions for different control tags in the labeling config.
        """
        control_models = []

        for control in self.label_interface.controls:
            # skipping tags without toName
            if not control.to_name:
                logger.warning(
                    f'{control.tag} {control.name} has no "toName" attribute, skipping it'
                )
                continue

            # match control tag with available control models
            for model_class in available_model_classes:
                if model_class.is_control_matched(control):
                    instance = model_class.create(self, control)
                    if not instance:
                        logger.debug(
                            f"No instance created for {control.tag} {control.name}"
                        )
                        continue
                    if not instance.label_map:
                        logger.error(
                            f"No label map built for the '{control.tag}' control tag '{instance.from_name}'.\n"
                            f"This indicates that your Label Studio config labels do not match the model's labels.\n"
                            f"To fix this, ensure that the 'value' or 'predicted_values' attribute "
                            f"in your Label Studio config matches one or more of these model labels.\n"
                            f"If you don't want to use this control tag for predictions, "
                            f'add `model_skip="true"` to it.\n'
                            f"Examples:\n"
                            f'  <Label value="Car"/>\n'
                            f'  <Label value="YourLabel" predicted_values="label1,label2"/>\n'
                            f"Labels provided in your labeling config:\n"
                            f"  {str(control.labels_attrs)}\n"
                            f"Available '{instance.model_path}' model labels:\n"
                            f"  {list(instance.model.names.values())}"
                        )
                        continue

                    control_models.append(instance)
                    logger.debug(f"Control tag with model detected: {instance}")
                    break

        if not control_models:
            control_tags = ", ".join([c.type for c in available_model_classes])
            raise ValueError(
                f"No suitable control tags (e.g. {control_tags} connected to Image or Video object tags) "
                f"detected in the label config:\n{self.label_config}"
            )

        return control_models

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Run YOLO predictions on the tasks, with optional Grounding DINO fallback
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions [Predictions array in JSON format]
            (https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(
            f"Run prediction on {len(tasks)} tasks, project ID = {self.project_id}"
        )

        # Check if hybrid mode is enabled
        use_hybrid = os.getenv("USE_HYBRID_MODE", "false").lower() in ["1", "true"]

        if use_hybrid and self.grounding_dino_model:
            return self._predict_hybrid(tasks, context, **kwargs)
        else:
            return self._predict_yolo_only(tasks, context, **kwargs)

    def _predict_yolo_only(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Original YOLO-only prediction logic"""
        control_models = self.detect_control_models()

        predictions = []
        for task in tasks:
            regions = []
            for model in control_models:
                path = model.get_path(task)
                regions += model.predict_regions(path)

            # calculate final score
            all_scores = [region["score"] for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # compose final prediction
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": self.model_version,
            }
            predictions.append(prediction)

        return ModelResponse(predictions=predictions)

    def _predict_hybrid(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Hybrid prediction using YOLO for known classes and Grounding DINO for unknown classes"""
        control_models = self.detect_control_models()

        predictions = []
        for task_idx, task in enumerate(tasks):
            logger.debug(f"Processing task {task_idx + 1}/{len(tasks)} (ID: {task.get('id', 'unknown')})")

            regions = []

            # First, try YOLO predictions
            yolo_regions = []
            for model in control_models:
                path = model.get_path(task)
                yolo_regions += model.predict_regions(path)

            # Get labels that YOLO has mappings for (vs labels that need Grounding DINO)
            yolo_mapped_labels = set()
            for model in control_models:
                if hasattr(model, 'label_map') and model.label_map:
                    # Labels that YOLO can map to (have predicted_values or exact matches)
                    yolo_mapped_labels.update(model.label_map.values())

            regions.extend(yolo_regions)

            # Use Grounding DINO for labels that YOLO cannot map (no predicted_values)
            if yolo_regions:  # Only run Grounding DINO if we have some YOLO results to potentially augment
                grounding_regions = self._predict_grounding_dino(task, yolo_mapped_labels)
                regions.extend(grounding_regions)
                if grounding_regions:
                    logger.info(f"Added {len(grounding_regions)} Grounding DINO detections to task {task.get('id', 'unknown')}")
            else:
                # If no YOLO predictions, we might still want to run Grounding DINO for all labels
                grounding_regions = self._predict_grounding_dino(task, yolo_mapped_labels)
                regions.extend(grounding_regions)

            # calculate final score
            all_scores = [region["score"] for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # compose final prediction
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": f"{self.model_version}_hybrid",
            }
            predictions.append(prediction)

        return ModelResponse(predictions=predictions)

    def _predict_grounding_dino(self, task: Dict, yolo_mapped_labels: set) -> List[Dict]:
        """Use Grounding DINO to predict additional classes not handled by YOLO"""
        if not self.grounding_dino_model:
            return []

        regions = []

        # Get control models to determine which ones support images
        control_models = self.detect_control_models()

        # Find RectangleLabels control for Grounding DINO
        rectangle_control = None
        for model in control_models:
            if hasattr(model, 'type') and model.type == 'RectangleLabels':
                rectangle_control = model
                break

        if not rectangle_control:
            logger.warning("No RectangleLabels control found for Grounding DINO predictions")
            return []

        # Get labels that need Grounding DINO (not handled by YOLO)
        all_labels = set()
        if rectangle_control.label_map:
            all_labels = set(rectangle_control.label_map.values())

        # Labels that need Grounding DINO are those not mapped by YOLO
        grounding_labels = all_labels - yolo_mapped_labels

        if not grounding_labels:
            logger.debug("No additional labels to predict with Grounding DINO")
            return []

        try:
            # Get image path
            path = rectangle_control.get_path(task)

            # Load image once
            src, img = load_image(path)
            H, W = src.shape[:2]

            # Always use batch mode: Process all labels in one prompt (faster)
            regions.extend(self._predict_grounding_dino_batch(img, grounding_labels, rectangle_control, H, W))

        except Exception as e:
            logger.error(f"Grounding DINO prediction failed: {e}")

        return regions


    def _predict_grounding_dino_batch(self, img, grounding_labels, rectangle_control, H, W):
        """Process all labels in one batch prompt with comma-separated labels"""
        regions = []

        # Create comma-separated prompt from all unmapped labels
        prompt = ", ".join(grounding_labels)
        logger.info(f"Grounding DINO batch prediction with prompt: {prompt}")

        try:
            # Run single Grounding DINO prediction for all labels at once
            # Grounding DINO predict() returns: boxes, logits, phrases
            boxes, logits, phrases = predict(
                model=self.grounding_dino_model,
                image=img,
                caption=prompt,  # Comma-separated: "boat, plane, ship, car"
                box_threshold=self.grounding_dino_box_threshold,
                text_threshold=self.grounding_dino_text_threshold,
                device=self.grounding_dino_device
            )

            logger.debug(f"Grounding DINO found {len(boxes)} detections with phrases: {phrases}")

            if len(boxes) == 0:
                logger.debug("No detections found for batch prompt")
                return regions

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            points = boxes_xyxy.cpu().numpy()

            # Grounding DINO returns phrases for each detection!
            # phrases is a list of strings, one for each detection
            logger.info(f"✅ Grounding DINO returned {len(phrases)} phrases: {phrases}")

            # Process each detection with its corresponding phrase
            for point, logit, phrase in zip(points, logits, phrases):
                score = float(logit.item())
                if score < rectangle_control.model_score_threshold:
                    continue

                # Clean the phrase (remove extra spaces, punctuation)
                clean_phrase = phrase.strip().lower().replace('.', '')
                logger.debug(f"Processing phrase: '{phrase}' -> cleaned: '{clean_phrase}'")

                # Try to match the phrase to our grounding labels
                matched_label = None
                for original_label in grounding_labels:
                    # Check if the phrase contains the label or vice versa
                    original_lower = original_label.lower()
                    if (original_lower in clean_phrase or
                        clean_phrase in original_lower or
                        clean_phrase == original_lower):
                        matched_label = original_label
                        break

                if matched_label:
                    output_label = rectangle_control.label_map.get(matched_label, matched_label) if rectangle_control.label_map else matched_label
                    logger.debug(f"✅ Phrase '{phrase}' matched to label '{matched_label}' -> '{output_label}'")
                else:
                    # Fallback to generic label if no match found
                    output_label = rectangle_control.label_map.get("detected_object", "Detected Object") if rectangle_control.label_map else "Detected Object"
                    logger.debug(f"⚠️ Phrase '{phrase}' not matched to any label, using generic '{output_label}'")

                region = {
                    "from_name": rectangle_control.from_name,
                    "to_name": rectangle_control.to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [output_label],
                        "x": (point[0] / W) * 100,
                        "y": (point[1] / H) * 100,
                        "width": ((point[2] - point[0]) / W) * 100,
                        "height": ((point[3] - point[1]) / H) * 100,
                    },
                    "score": score,
                }
                regions.append(region)

            # Convert all detections to Label Studio format with generic label
            for point, logit in zip(points, logits):
                score = float(logit.item())
                if score < rectangle_control.model_score_threshold:
                    continue

                region = {
                    "from_name": rectangle_control.from_name,
                    "to_name": rectangle_control.to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [output_label],
                        "x": (point[0] / W) * 100,
                        "y": (point[1] / H) * 100,
                        "width": ((point[2] - point[0]) / W) * 100,
                        "height": ((point[3] - point[1]) / H) * 100,
                    },
                    "score": score,
                }
                regions.append(region)

            logger.info(f"Grounding DINO found {len(regions)} detections for {len(grounding_labels)} labels")

        except Exception as e:
            logger.error(f"Grounding DINO batch prediction failed: {e}")

        return regions

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated.
        Or it's called when "Start training" clicked on the model in the project settings.
        """
        results = {}
        control_models = self.detect_control_models()
        for model in control_models:
            training_result = model.fit(event, data, **kwargs)
            results[model.from_name] = training_result

        return results
