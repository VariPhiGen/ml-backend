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

# Module-level cache for OWL-ViT model (shared across all instances)
_owl_vit_cache = {
    'model': None,
    'processor': None,
    'device': None,
    'box_threshold': None,
    'text_threshold': None,
}

logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

# OWL-ViT availability is checked by _check_owl_vit_available() function

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

        # Initialize OWL-ViT if available (for hybrid mode)
        # This happens during container startup, so be defensive
        try:
            if _check_owl_vit_available():
                self._init_owl_vit()
            else:
                logger.warning("OWL-ViT dependencies not available - hybrid mode disabled")
                self.owl_vit_model = None
                self.owl_vit_processor = None
        except Exception as e:
            logger.error(f"Failed to initialize OWL-ViT during startup: {e}")
            logger.warning("Continuing without hybrid mode - YOLO-only predictions available")
            self.owl_vit_model = None
            self.owl_vit_processor = None

    def _init_owl_vit(self):
        """Initialize OWL-ViT model (uses module-level cache to avoid reloading)"""
        # Check module-level cache first to avoid reloading on every request
        if _owl_vit_cache['model'] is not None:
            logger.info("Reusing cached OWL-ViT model (already loaded in this worker)")
            self.owl_vit_model = _owl_vit_cache['model']
            self.owl_vit_processor = _owl_vit_cache['processor']
            self.owl_vit_device = _owl_vit_cache['device']
            self.owl_vit_box_threshold = _owl_vit_cache['box_threshold']
            self.owl_vit_text_threshold = _owl_vit_cache['text_threshold']
            return
        
        try:
            import torch
            from transformers import OwlViTProcessor, OwlViTForObjectDetection

            logger.info("Loading OWL-ViT model (this may take a moment on first run)...")

            # Load with local cache and error handling
            processor = OwlViTProcessor.from_pretrained(
                "google/owlvit-base-patch32",
                cache_dir=os.getenv("TRANSFORMERS_CACHE", "/tmp/transformers_cache")
            )
            model = OwlViTForObjectDetection.from_pretrained(
                "google/owlvit-base-patch32",
                cache_dir=os.getenv("TRANSFORMERS_CACHE", "/tmp/transformers_cache")
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            box_threshold = float(os.getenv("OWL_VIT_BOX_THRESHOLD", "0.1"))
            text_threshold = float(os.getenv("OWL_VIT_TEXT_THRESHOLD", "0.0"))

            # Store in module-level cache
            _owl_vit_cache['model'] = model
            _owl_vit_cache['processor'] = processor
            _owl_vit_cache['device'] = device
            _owl_vit_cache['box_threshold'] = box_threshold
            _owl_vit_cache['text_threshold'] = text_threshold

            # Also set on instance
            self.owl_vit_model = model
            self.owl_vit_processor = processor
            self.owl_vit_device = device
            self.owl_vit_box_threshold = box_threshold
            self.owl_vit_text_threshold = text_threshold

            logger.info(f"✅ OWL-ViT initialized successfully on device: {device}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OWL-ViT: {e}")
            logger.error("This might be due to network issues or missing dependencies")
            logger.error("Hybrid mode will not work without OWL-ViT")
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
                        # Empty label_map is OK for OWL-ViT - these controls will be handled by OWL-ViT
                        # Log as warning (not error) since this is expected for controls with labels
                        # that don't match YOLO's classes (e.g., PPE labels like helmet, vest)
                        logger.warning(
                            f"No YOLO label map for '{control.tag}' control '{instance.from_name}'.\n"
                            f"This is OK - labels in this control will be handled by OWL-ViT if hybrid mode is enabled.\n"
                            f"Labels in this control: {list(control.labels_attrs.keys()) if control.labels_attrs else 'N/A'}\n"
                            f"YOLO model labels: {list(instance.model.names.values())[:10]}..."
                        )
                        # DON'T skip - we need this control for OWL-ViT even if YOLO can't map its labels
                        # Only skip if model_skip="true" is set (handled above)

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

        if use_hybrid and self.owl_vit_model:
            return self._predict_hybrid(tasks, context, **kwargs)
        else:
            return self._predict_yolo_only(tasks, context, **kwargs)

    def _predict_yolo_only(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Original YOLO-only prediction logic"""
        control_models = self.detect_control_models()

        predictions = []
        for task in tasks:
            regions = []
            model_files_used = []
            for model in control_models:
                path = model.get_path(task)
                regions += model.predict_regions(path)
                # Track which model file was used
                model_file = getattr(model, 'model_path', 'unknown')
                if model_file not in model_files_used:
                    model_files_used.append(model_file)

            # calculate final score
            all_scores = [region["score"] for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # compose final prediction
            model_file = model_files_used[0] if model_files_used else "unknown"
            logger.info(f"Using model file(s): {model_files_used}")
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": self.model_version,
                "model_file": model_file,
            }
            if len(model_files_used) > 1:
                prediction["model_files"] = model_files_used
            logger.debug(f"Prediction keys: {list(prediction.keys())}")
            logger.info(f"Prediction dict includes model_file: {'model_file' in prediction}, value: {prediction.get('model_file')}")
            predictions.append(prediction)

        logger.info(f"Returning ModelResponse with {len(predictions)} predictions")
        response = ModelResponse(predictions=predictions)
        logger.info(f"After ModelResponse creation, first prediction keys: {list(response.model_dump()['predictions'][0].keys()) if response.model_dump().get('predictions') else 'NO PREDICTIONS'}")
        return response

    def _predict_hybrid(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Hybrid prediction using YOLO for known classes and Grounding DINO for unknown classes"""
        control_models = self.detect_control_models()

        predictions = []
        for task_idx, task in enumerate(tasks):
            logger.debug(f"Processing task {task_idx + 1}/{len(tasks)} (ID: {task.get('id', 'unknown')})")

            regions = []

            # First, try YOLO predictions
            yolo_regions = []
            model_files_used = []
            for model in control_models:
                path = model.get_path(task)
                yolo_regions += model.predict_regions(path)
                # Track which model file was used
                model_file = getattr(model, 'model_path', 'unknown')
                if model_file not in model_files_used:
                    model_files_used.append(model_file)

            # Get labels that YOLO has mappings for (vs labels that need Grounding DINO)
            yolo_mapped_labels = set()
            for model in control_models:
                if hasattr(model, 'label_map') and model.label_map:
                    # Labels that YOLO can map to (have predicted_values or exact matches)
                    yolo_mapped_labels.update(model.label_map.values())

            regions.extend(yolo_regions)
            logger.info(f"YOLO found {len(yolo_regions)} detections for task {task.get('id', 'unknown')}")
            
            # Log YOLO region structure for comparison
            if yolo_regions:
                import json
                logger.info(f"Sample YOLO region structure: {json.dumps(yolo_regions[0], indent=2, default=str)}")

            # Use OWL-ViT for labels that YOLO cannot map (no predicted_values)
            # Always run OWL-ViT (even if YOLO found nothing) to detect unmapped labels
            owl_regions = self._predict_owl_vit(task, yolo_mapped_labels)
            if owl_regions:
                logger.info(f"Adding {len(owl_regions)} OWL-ViT detections to task {task.get('id', 'unknown')}")
                # Log OWL-ViT region structure for comparison - should match YOLO exactly
                import json
                logger.info(f"Sample OWL-ViT region structure: {json.dumps(owl_regions[0], indent=2, default=str)}")
                
                # Compare structures to ensure they're identical
                if yolo_regions and owl_regions:
                    yolo_keys = set(yolo_regions[0].keys())
                    owl_keys = set(owl_regions[0].keys())
                    if yolo_keys != owl_keys:
                        logger.warning(f"⚠️ Structure mismatch! YOLO keys: {yolo_keys}, OWL-ViT keys: {owl_keys}")
                    else:
                        logger.info(f"✅ Region structures match: {yolo_keys}")
                    
                    yolo_value_keys = set(yolo_regions[0].get('value', {}).keys())
                    owl_value_keys = set(owl_regions[0].get('value', {}).keys())
                    if yolo_value_keys != owl_value_keys:
                        logger.warning(f"⚠️ Value structure mismatch! YOLO value keys: {yolo_value_keys}, OWL-ViT value keys: {owl_value_keys}")
                    else:
                        logger.info(f"✅ Value structures match: {yolo_value_keys}")
                
                regions.extend(owl_regions)
            else:
                logger.info(f"No OWL-ViT detections for task {task.get('id', 'unknown')}")

            # calculate final score
            all_scores = [region["score"] for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # Log region details for debugging
            logger.info(f"Total regions for task {task.get('id', 'unknown')}: {len(regions)} (YOLO: {len(yolo_regions)}, OWL-ViT: {len(owl_regions) if 'owl_regions' in locals() else 0})")
            if regions:
                logger.info(f"Sample region labels: {[r.get('value', {}).get('rectanglelabels', []) for r in regions[:3]]}")
                # Log full structure of first OWL-ViT region for debugging
                owl_vit_regions = [r for r in regions if r not in yolo_regions]
                if owl_vit_regions:
                    import json
                    logger.info(f"First OWL-ViT region structure: {json.dumps(owl_vit_regions[0], indent=2, default=str)}")

            # compose final prediction
            # Log the exact order and content of regions before creating prediction
            logger.info(f"Creating prediction with {len(regions)} total regions")
            for idx, region in enumerate(regions):
                region_label = region.get('value', {}).get('rectanglelabels', ['unknown'])[0] if region.get('value', {}).get('rectanglelabels') else 'unknown'
                region_score = region.get('score', 'N/A')
                logger.debug(f"Region {idx+1} in final list: label={region_label}, score={region_score}")
            
            # Ensure all numeric values are proper types for JSON serialization
            # IMPORTANT: regions list should contain both YOLO and OWL-ViT regions in order
            prediction = {
                "result": regions,  # This is the merged list: YOLO regions first, then OWL-ViT regions
                "score": float(avg_score) if avg_score is not None else 0.0,
                "model_version": f"{self.model_version}_hybrid",
                "model_file": model_files_used[0] if model_files_used else "unknown",
            }
            if len(model_files_used) > 1:
                prediction["model_files"] = model_files_used
            if self.owl_vit_model:
                prediction["hybrid_mode"] = True
                prediction["owl_vit_enabled"] = True
            
            # Verify the prediction structure before appending
            logger.info(f"Prediction structure: result has {len(prediction['result'])} regions")
            if prediction['result']:
                result_labels = [r.get('value', {}).get('rectanglelabels', ['unknown'])[0] if r.get('value', {}).get('rectanglelabels') else 'unknown' for r in prediction['result']]
                logger.info(f"Prediction result labels in order: {result_labels}")
            
            predictions.append(prediction)
            
            # Log final prediction summary
            logger.info(f"Final prediction for task {task.get('id', 'unknown')}: {len(regions)} regions, score={avg_score:.3f}")

        logger.info(f"Returning {len(predictions)} predictions from hybrid mode")
        response = ModelResponse(predictions=predictions)
        
        # Log the serialized response for debugging
        try:
            dumped = response.model_dump()
            logger.info(f"ModelResponse serialized: {len(dumped.get('predictions', []))} predictions")
            if dumped.get('predictions'):
                first_pred = dumped['predictions'][0]
                regions = first_pred.get('result', [])
                logger.info(f"First prediction has {len(regions)} regions")
                
                # Count YOLO vs OWL-ViT regions
                yolo_count = len(yolo_regions) if 'yolo_regions' in locals() else 0
                owl_count = len(owl_regions) if 'owl_regions' in locals() else 0
                logger.info(f"Region breakdown: {yolo_count} YOLO + {owl_count} OWL-ViT = {len(regions)} total")
                
                if regions:
                    # Log all regions to see YOLO vs OWL-ViT
                    for idx, region in enumerate(regions):
                        region_type = region.get('type')
                        labels = region.get('value', {}).get('rectanglelabels', [])
                        score = region.get('score', 'N/A')
                        from_name = region.get('from_name')
                        to_name = region.get('to_name')
                        value = region.get('value', {})
                        x = value.get('x', 'N/A')
                        y = value.get('y', 'N/A')
                        w = value.get('width', 'N/A')
                        h = value.get('height', 'N/A')
                        
                        # Format score and coordinates safely
                        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                        x_str = f"{x:.2f}" if isinstance(x, (int, float)) else str(x)
                        y_str = f"{y:.2f}" if isinstance(y, (int, float)) else str(y)
                        w_str = f"{w:.2f}" if isinstance(w, (int, float)) else str(w)
                        h_str = f"{h:.2f}" if isinstance(h, (int, float)) else str(h)
                        
                        # Determine if this is YOLO or OWL-ViT based on label
                        source = "YOLO" if labels and labels[0] in yolo_mapped_labels else "OWL-ViT"
                        logger.info(f"Region {idx+1} [{source}]: type={region_type}, label={labels}, score={score_str}, "
                                   f"from_name={from_name}, to_name={to_name}, "
                                   f"coords=(x={x_str}, y={y_str}, w={w_str}, h={h_str})")
        except Exception as e:
            logger.warning(f"Could not serialize response for logging: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return response

    def _predict_owl_vit(self, task: Dict, yolo_mapped_labels: set) -> List[Dict]:
        """Use OWL-ViT to predict additional classes not handled by YOLO"""
        if not self.owl_vit_model or not self.owl_vit_processor:
            return []

        regions = []

        # Get control models to determine which ones support images
        control_models = self.detect_control_models()

        # Find ALL RectangleLabels controls (there may be multiple with different names)
        rectangle_controls = []
        for model in control_models:
            if hasattr(model, 'type') and model.type == 'RectangleLabels':
                rectangle_controls.append(model)

        if not rectangle_controls:
            logger.warning("No RectangleLabels control found for OWL-ViT predictions")
            return []

        # Build a DYNAMIC mapping of label -> control based on Label Studio config
        # This ensures each label is assigned to its correct control (from_name) regardless of
        # which model (YOLO or OWL-ViT) produces it. This prevents "overshading" or render rule
        # issues in Label Studio where predictions must match their control's from_name.
        #
        # IMPORTANT: OWL-ViT can detect ANY label, including "car" or "person" if they're not
        # in YOLO's label_map. The from_name is determined by which CONTROL the label belongs
        # to in the config, NOT by which model detects it.
        # Example: If "car" is in "bbox" control but not in YOLO's label_map, OWL-ViT will
        # detect it and use from_name="bbox" (not "bbox_ppe").
        label_to_control = {}
        all_owl_labels = set()
        
        for rectangle_control in rectangle_controls:
            logger.info(f"Processing control: from_name='{rectangle_control.from_name}', to_name='{rectangle_control.to_name}'")
            
            # Get labels defined in this control from the Label Studio config
            control_labels = set()
            if hasattr(rectangle_control, 'control') and rectangle_control.control:
                if hasattr(rectangle_control.control, 'labels_attrs') and rectangle_control.control.labels_attrs:
                    # labels_attrs contains all <Label> tags defined in this control
                    control_labels = set(rectangle_control.control.labels_attrs.keys())
                    logger.info(f"Control '{rectangle_control.from_name}' has {len(control_labels)} labels from labels_attrs: {control_labels}")
                else:
                    logger.warning(f"Control '{rectangle_control.from_name}' has no labels_attrs")
            else:
                logger.warning(f"Control '{rectangle_control.from_name}' has no control attribute")
            
            # Fallback: if we can't get labels from control, use label_map values
            if not control_labels and rectangle_control.label_map:
                control_labels = set(rectangle_control.label_map.values())
                logger.info(f"Control '{rectangle_control.from_name}' using {len(control_labels)} labels from label_map: {control_labels}")
            
            if not control_labels:
                logger.warning(f"Control '{rectangle_control.from_name}' has NO labels found!")
            
            # For each label in this control that's NOT in YOLO's label_map, assign it to this control
            # Labels in yolo_mapped_labels are handled by YOLO (which already uses correct from_name).
            # Labels NOT in yolo_mapped_labels will be handled by OWL-ViT, and must use the correct
            # from_name based on which control they belong to in the config.
            for label in control_labels:
                if label not in yolo_mapped_labels:
                    # Label not in YOLO's label_map → OWL-ViT will handle it
                    # Assign to the control where this label is defined (prevents overshading)
                    label_to_control[label] = rectangle_control
                    all_owl_labels.add(label)
                    logger.info(f"Label '{label}' → control '{rectangle_control.from_name}' (OWL-ViT, not in YOLO label_map)")
                else:
                    logger.debug(f"Label '{label}' skipped (in YOLO label_map, handled by YOLO)")

        logger.info(f"YOLO mapped labels: {yolo_mapped_labels}")
        logger.info(f"OWL-ViT labels found: {all_owl_labels}")
        logger.info(f"Label-to-control mapping: {[(label, ctrl.from_name) for label, ctrl in label_to_control.items()]}")

        if not all_owl_labels:
            logger.debug("No additional labels to predict with OWL-ViT")
            return []

        try:
            # Group labels by their control (since different controls may have different from_name/to_name)
            # Use from_name as key since control objects are not hashable
            control_to_labels = {}
            for label, control in label_to_control.items():
                control_key = control.from_name  # Use from_name as hashable key
                if control_key not in control_to_labels:
                    control_to_labels[control_key] = {'control': control, 'labels': []}
                control_to_labels[control_key]['labels'].append(label)
            
            logger.info(f"Processing {len(control_to_labels)} control(s) with OWL-ViT labels")
            
            # Use the first control to get the image path (all controls should point to the same image)
            first_control_key = list(control_to_labels.keys())[0]
            first_control = control_to_labels[first_control_key]['control']
            path = first_control.get_path(task)

            # Load image once (shared across all controls)
            from PIL import Image
            image = Image.open(path).convert("RGB")
            # PIL image.size returns (width, height)
            img_width, img_height = image.size

            # Process each control's labels separately to ensure correct from_name/to_name
            for control_key, control_data in control_to_labels.items():
                rectangle_control = control_data['control']
                owl_labels = control_data['labels']
                logger.info(f"Processing {len(owl_labels)} OWL-ViT labels for control '{rectangle_control.from_name}': {owl_labels}")
                control_regions = self._predict_owl_vit_batch(image, owl_labels, rectangle_control, img_width, img_height)
                regions.extend(control_regions)
                logger.info(f"Control '{rectangle_control.from_name}' produced {len(control_regions)} regions")

        except Exception as e:
            logger.error(f"OWL-ViT prediction failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        return regions


    def _predict_owl_vit_batch(self, image, owl_labels, rectangle_control, img_width, img_height):
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
            # Try using normalized coordinates directly (without target_sizes)
            # This avoids potential coordinate system confusion
            results = self.owl_vit_processor.post_process_object_detection(
                outputs=outputs, threshold=self.owl_vit_box_threshold
            )

            # Process detections
            logger.info(f"OWL-ViT using score threshold: {rectangle_control.model_score_threshold}")
            for i, (scores, labels, boxes) in enumerate(zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"])):
                score = float(scores.item())
                if score < rectangle_control.model_score_threshold:
                    logger.debug(f"OWL-ViT detection {i+1} filtered: score {score:.3f} < threshold {rectangle_control.model_score_threshold}")
                    continue
                logger.debug(f"OWL-ViT detection {i+1} passed threshold: score {score:.3f} >= {rectangle_control.model_score_threshold}")

                # Get the corresponding label from our queries
                label_idx = int(labels.item())
                if label_idx < len(owl_labels):
                    detected_label = list(owl_labels)[label_idx]
                    # OWL-ViT labels are already Label Studio label names, so use directly
                    # label_map is for YOLO labels, so we don't need to map OWL-ViT labels
                    output_label = detected_label
                    
                    # Verify the label exists in the Label Studio config
                    label_valid = True
                    if hasattr(rectangle_control, 'control') and rectangle_control.control:
                        if hasattr(rectangle_control.control, 'labels_attrs'):
                            available_labels = set(rectangle_control.control.labels_attrs.keys())
                            if output_label not in available_labels:
                                logger.warning(f"⚠️ OWL-ViT label '{output_label}' NOT FOUND in Label Studio config!")
                                logger.warning(f"   Available labels: {sorted(available_labels)}")
                                logger.warning(f"   This prediction will be REJECTED by Label Studio")
                                # Try to find a case-insensitive match
                                label_lower = output_label.lower()
                                matched = None
                                for ls_label in available_labels:
                                    if ls_label.lower() == label_lower:
                                        matched = ls_label
                                        break
                                if matched:
                                    logger.info(f"✅ Using case-insensitive match: '{output_label}' -> '{matched}'")
                                    output_label = matched
                                    label_valid = True
                                else:
                                    logger.error(f"❌ Skipping detection - label '{output_label}' not in config and no match found")
                                    label_valid = False
                            else:
                                logger.info(f"✅ OWL-ViT label '{output_label}' validated - exists in Label Studio config")
                    
                    if not label_valid:
                        continue  # Skip this detection
                    
                    logger.debug(f"OWL-ViT detected label index {label_idx}: '{detected_label}' -> '{output_label}' (validated)")
                else:
                    logger.warning(f"OWL-ViT label index {label_idx} out of range (max: {len(owl_labels)-1})")
                    continue  # Skip instead of using placeholder

                # When target_sizes is NOT provided, post_process_object_detection returns 
                # boxes in normalized coordinates (0-1) in (x1, y1, x2, y2) format
                # where x1, y1, x2, y2 are all between 0 and 1
                box = boxes.cpu().numpy()
                x1_norm, y1_norm, x2_norm, y2_norm = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                
                # Log raw normalized coordinates for debugging
                logger.info(f"OWL-ViT detection {i+1}: normalized coords ({x1_norm:.4f}, {y1_norm:.4f}, {x2_norm:.4f}, {y2_norm:.4f}), "
                           f"image size: {img_width}x{img_height}")
                
                # Clamp normalized coordinates to [0, 1]
                x1_norm = max(0.0, min(1.0, x1_norm))
                y1_norm = max(0.0, min(1.0, y1_norm))
                x2_norm = max(0.0, min(1.0, x2_norm))
                y2_norm = max(0.0, min(1.0, y2_norm))
                
                # Ensure valid box
                if x2_norm <= x1_norm or y2_norm <= y1_norm:
                    logger.warning(f"Invalid normalized box: ({x1_norm}, {y1_norm}, {x2_norm}, {y2_norm})")
                    continue
                
                # Convert from normalized coordinates (0-1) directly to Label Studio percentage format
                # Match YOLO exactly: no rounding, direct calculation like YOLO does
                # Label Studio expects: x = left edge %, y = top edge %, width = width %, height = height %
                x_percent = x1_norm * 100  # Match YOLO - no rounding
                y_percent = y1_norm * 100
                width_percent = (x2_norm - x1_norm) * 100
                height_percent = (y2_norm - y1_norm) * 100
                
                # Validate coordinates are within bounds (0-100%)
                if x_percent < 0 or x_percent > 100 or y_percent < 0 or y_percent > 100:
                    logger.warning(f"OWL-ViT detection {i+1}: coordinates out of bounds, skipping")
                    continue
                if width_percent <= 0 or height_percent <= 0:
                    logger.warning(f"OWL-ViT detection {i+1}: invalid dimensions, skipping")
                    continue
                if x_percent + width_percent > 100.01:  # Allow small floating point errors
                    logger.warning(f"OWL-ViT detection {i+1}: box extends beyond image (x+w={x_percent+width_percent:.2f}%), clamping")
                    width_percent = 100 - x_percent
                if y_percent + height_percent > 100.01:  # Allow small floating point errors
                    logger.warning(f"OWL-ViT detection {i+1}: box extends beyond image (y+h={y_percent+height_percent:.2f}%), clamping")
                    height_percent = 100 - y_percent
                
                # Ensure width and height are positive after clamping
                if width_percent <= 0 or height_percent <= 0:
                    logger.warning(f"OWL-ViT detection {i+1}: invalid dimensions after clamping, skipping")
                    continue
                
                # Match YOLO region structure EXACTLY - same field order, types, and calculation style
                # CRITICAL: Ensure this matches YOLO's rectangle_labels.py structure exactly
                region = {
                    "from_name": rectangle_control.from_name,
                    "to_name": rectangle_control.to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [output_label],  # Must be a list with one string
                        "x": x_percent,  # Match YOLO exactly - direct calculation, no rounding
                        "y": y_percent,
                        "width": width_percent,
                        "height": height_percent,
                    },
                    "score": score,  # Match YOLO exactly - direct value, no float() conversion
                }
                
                # Verify region structure matches YOLO exactly
                required_keys = {"from_name", "to_name", "type", "value", "score"}
                if set(region.keys()) != required_keys:
                    logger.error(f"OWL-ViT region missing required keys! Has: {set(region.keys())}, Needs: {required_keys}")
                value_keys = {"rectanglelabels", "x", "y", "width", "height"}
                if set(region["value"].keys()) != value_keys:
                    logger.error(f"OWL-ViT region value missing keys! Has: {set(region['value'].keys())}, Needs: {value_keys}")
                
                logger.info(f"OWL-ViT detection {i+1}: label='{output_label}', score={score:.3f}, "
                           f"from_name='{rectangle_control.from_name}', to_name='{rectangle_control.to_name}', "
                           f"coords (x={x_percent:.2f}%, y={y_percent:.2f}%, w={width_percent:.2f}%, h={height_percent:.2f}%)")
                
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
