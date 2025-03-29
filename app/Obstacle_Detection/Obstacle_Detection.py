import cv2
from app.Obstacle_Detection.DistanceAlgorithm import DistanceAlgorithm
from app.Obstacle_Detection.Zone import Zone
import json

class ObstacleDetection:

    def __init__(self):
        json_file = open('/code/app/Obstacle_Detection/settings.json')
        jsonFileData = json.load(json_file)
        inputSettings = jsonFileData["input_settings"]
        settings = jsonFileData["obstacle_detection_settings"]
        frame_width = int(inputSettings["frame_width"])
        frame_height = int(inputSettings["frame_height"])

        self.zone = Zone(settings["zone_settings"], frame_width, frame_height)
        self.draw_zones = settings["draw_zones"]

        self.distanceAlgorithm = DistanceAlgorithm(settings["distance_algorithm"])
        self.model = None


    def produce_outputOld(self, frame, model): # Text döndürmek için aşağıdaki produce output fonksiyonu kullanılıyor
        self.model = model
        results = self.model(frame)
        detections = results[0]

        for det in detections.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, det.xyxy[0])  # Convert to integers
            conf = det.conf[0]  # Confidence score
            cls = int(det.cls[0])  # Class ID
            class_name = self.model.names[cls]  # Class name

            color = self.zone.get_bbox_color((x1, y1, x2, y2))

            if color:
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                distance = self.distanceAlgorithm.calculate(det, class_name)

                # Display the distance
                cv2.putText(frame, f"Distance: {distance / 100:.1f} m", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Print detected object in the red area
                if color == (0, 0, 255):
                    print(f"Önünde {class_name} var, Uzaklığı {round(distance / 100)} metre")

        return frame

    def produce_output(self, frame, model):
        self.model = model
        results = self.model(frame)
        detections = results[0]
        height, width, _ = frame.shape

        min_distance = float('inf')
        closest_object_text = ""

        for det in detections.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, det.xyxy[0])  # Convert to integers
            conf = det.conf[0]  # Confidence score
            cls = int(det.cls[0])  # Class ID
            class_name = self.model.names[cls]  # Class name

            color = self.zone.get_bbox_color((x1, y1, x2, y2))

            if color:
                distance = self.distanceAlgorithm.calculate(det, class_name)

                # Update the closest object text if the distance is smaller
                if distance < min_distance:
                    min_distance = distance
                    closest_object_text = f"{round(distance / 100)} metre önünde {class_name} var"

        return closest_object_text