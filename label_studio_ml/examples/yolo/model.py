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

# Test imports for hybrid mode
def _check_owl_vit_available():
    """Check if OWL-ViT is available for import"""
    try:
        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        return True
    except ImportError:
        return False


# Initial check
OWL_VIT_AVAILABLE = _check_owl_vit_available()


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

        # Initialize OWL-ViT if available
        if _check_owl_vit_available():
            self._init_owl_vit()
        else:
            logger.warning("OWL-ViT not available - hybrid mode will not work")
            self.owl_vit_model = None
            self.owl_vit_processor = None

    def _init_owl_vit(self):
        """Initialize OWL-ViT model"""
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection

            self.owl_vit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.owl_vit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

            self.owl_vit_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.owl_vit_model.to(self.owl_vit_device)

            self.owl_vit_box_threshold = float(os.getenv("OWL_VIT_BOX_THRESHOLD", "0.1"))
            self.owl_vit_text_threshold = float(os.getenv("OWL_VIT_TEXT_THRESHOLD", "0.0"))

            logger.info(f"OWL-ViT initialized on device: {self.owl_vit_device}")
        except Exception as e:
            logger.warning(f"Failed to initialize OWL-ViT: {e}")
            self.owl_vit_model = None
            self.owl_vit_processor = None

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

            # Use OWL-ViT for labels that YOLO cannot map (no predicted_values)
            if yolo_regions:  # Only run OWL-ViT if we have some YOLO results to potentially augment
                owl_regions = self._predict_owl_vit(task, yolo_mapped_labels)
                regions.extend(owl_regions)
                if owl_regions:
                    logger.info(f"Added {len(owl_regions)} OWL-ViT detections to task {task.get('id', 'unknown')}")
            else:
                # If no YOLO predictions, we might still want to run OWL-ViT for all labels
                owl_regions = self._predict_owl_vit(task, yolo_mapped_labels)
                regions.extend(owl_regions)

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

    def _predict_owl_vit(self, task: Dict, yolo_mapped_labels: set) -> List[Dict]:
        """Use OWL-ViT to predict additional classes not handled by YOLO"""
        if not self.owl_vit_model or not self.owl_vit_processor:
            return []

        regions = []

        # Get control models to determine which ones support images
        control_models = self.detect_control_models()

        # Find RectangleLabels control for OWL-ViT
        rectangle_control = None
        for model in control_models:
            if hasattr(model, 'type') and model.type == 'RectangleLabels':
                rectangle_control = model
                break

        if not rectangle_control:
            logger.warning("No RectangleLabels control found for OWL-ViT predictions")
            return []

        # Get labels that need OWL-ViT (not handled by YOLO)
        all_labels = set()
        if rectangle_control.label_map:
            all_labels = set(rectangle_control.label_map.values())

        # Labels that need OWL-ViT are those not mapped by YOLO
        owl_labels = all_labels - yolo_mapped_labels

        if not owl_labels:
            logger.debug("No additional labels to predict with OWL-ViT")
            return []

        try:
            # Get image path
            path = rectangle_control.get_path(task)

            # Load image
            from PIL import Image
            image = Image.open(path).convert("RGB")
            H, W = image.size

            # Process all labels with OWL-ViT
            regions.extend(self._predict_owl_vit_batch(image, owl_labels, rectangle_control, H, W))

        except Exception as e:
            logger.error(f"OWL-ViT prediction failed: {e}")

        return regions


    def _predict_owl_vit_batch(self, image, owl_labels, rectangle_control, H, W):
        """Process labels with OWL-ViT zero-shot detection"""
        regions = []

        # Convert labels to text queries
        text_queries = [f"a photo of a {label}" for label in owl_labels]
        logger.info(f"OWL-ViT prediction with queries: {text_queries}")

        try:
            import torch

            # Process image and text
            inputs = self.owl_vit_processor(text=text_queries, images=image, return_tensors="pt")
            inputs = {k: v.to(self.owl_vit_device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.owl_vit_model(**inputs)

            # Process results
            target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
            results = self.owl_vit_processor.post_process_object_detection(
                outputs=outputs, threshold=self.owl_vit_box_threshold, target_sizes=target_sizes
            )

            # Process detections
            for i, (scores, labels, boxes) in enumerate(zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"])):
                score = float(scores.item())
                if score < rectangle_control.model_score_threshold:
                    continue

                # Get the corresponding label from our queries
                label_idx = int(labels.item())
                if label_idx < len(owl_labels):
                    detected_label = list(owl_labels)[label_idx]
                    output_label = rectangle_control.label_map.get(detected_label, detected_label)
                else:
                    output_label = rectangle_control.label_map.get("detected_object", "Detected Object")

                # Convert boxes from (x1, y1, x2, y2) to percentage coordinates
                box = boxes.cpu().numpy()
                region = {
                    "from_name": rectangle_control.from_name,
                    "to_name": rectangle_control.to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [output_label],
                        "x": (box[0] / W) * 100,
                        "y": (box[1] / H) * 100,
                        "width": ((box[2] - box[0]) / W) * 100,
                        "height": ((box[3] - box[1]) / H) * 100,
                    },
                    "score": score,
                }
                regions.append(region)

            logger.info(f"OWL-ViT found {len(regions)} detections")

        except Exception as e:
            logger.error(f"OWL-ViT prediction error: {e}")

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
