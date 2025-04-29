import cv2
import torch
import os

from .DistanceAlgorithm import DistanceAlgorithm
from .Zone import Zone
import json


class ObstacleDetection:
    # Cache for settings to avoid reloading
    _settings_cache = None
    
    def __init__(self):
        # Load settings only once and cache them
        
        if ObstacleDetection._settings_cache is None:
            settings_path = 'app/Obstacle_Detection/settings.json'
            print("list of directories: ", os.listdir())
            try:
                with open(settings_path) as json_file:
                    print("settings_path: ", settings_path)
                    ObstacleDetection._settings_cache = json.load(json_file)
            except FileNotFoundError:
                # Attempt to find settings file in parent directories
                for _ in range(3):  # Try up to 3 parent directories
                    settings_path = os.path.join('..', settings_path)
                    if os.path.exists(settings_path):
                        with open(settings_path) as json_file:
                            ObstacleDetection._settings_cache = json.load(json_file)
                        break
            
            if ObstacleDetection._settings_cache is None:
                raise FileNotFoundError("Could not find settings.json file")
        
        jsonFileData = ObstacleDetection._settings_cache
        inputSettings = jsonFileData["input_settings"]
        settings = jsonFileData["obstacle_detection_settings"]
        frame_width = int(inputSettings["frame_width"])
        frame_height = int(inputSettings["frame_height"])

        self.zone = Zone(settings["zone_settings"], frame_width, frame_height)
        self.draw_zones = settings["draw_zones"]
        self.distanceAlgorithm = DistanceAlgorithm(settings["distance_algorithm"])
        self.model = None

    def produce_output(self, frame, model):
        """
        Process frame to detect objects and return list of objects with their details
        Args:
            frame: Input image frame
            model: YOLO model for detection
        Returns:
            List of dictionaries containing Object, Distance, and Danger level
        """
        self.model = model
        
        # Run inference with the model
        with torch.no_grad():
            results = self.model(frame)
        
        detections = results[0]
        print("detections: ", detections.boxes)
        objects_list = []
        class_names = []

        # Process each detection
        for det in detections.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, det.xyxy[0])  # Convert to integers
            conf = float(det.conf[0])  # Confidence score
            cls = int(det.cls[0])  # Class ID
            
            # Skip low confidence detections
            if conf < 0.4:  # Confidence threshold
                continue
                
            class_name = self.model.names[cls]  # Class name
            class_names.append(class_name)
            color = self.zone.get_bbox_color((x1, y1, x2, y2))

            if True:
                distance = self.distanceAlgorithm.calculate(det, class_name)
                distance_m = distance / 100

                # Determine danger level based on distance and zone
                if color == (0, 0, 255):  # Red zone
                    if distance_m < 1.5:
                        danger = "Red"
                    elif distance_m <= 3:
                        danger = "Yellow"
                    else:
                        danger = "Yellow"
                elif color == (0, 255, 255):  # Yellow zone
                    if distance_m < 1.5:
                        danger = "Yellow"
                    else:
                        danger = "Green"
                elif color == (0, 255, 0):  # Green zone
                    if distance_m < 1.5:
                        danger = "Yellow"
                    else:
                        danger = "Green"
                else:
                    danger = "Unknown"

                # Add to objects list
                objects_list.append({
                    "Object": class_name,
                    "Distance": round(distance_m, 2),
                    "Danger": danger,
                    "Confidence": round(conf, 2)
                })
                print("objects_list: ", objects_list)

        return objects_list, class_names
