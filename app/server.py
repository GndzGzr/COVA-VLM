import numpy as np
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
import pytesseract

from app.Obstacle_Detection.Obstacle_Detection import ObstacleDetection
from app.atm import produce_output

button_model = None
fingertip_model = None

def create_app():
    app = Flask(__name__)

    global button_model, fingertip_model


    button_model = YOLO("/code/bestLR.pt")
    fingertip_model = YOLO("/code/finger_detector.pt")
    WA_model = YOLO("/code/WA_model.pt")

    """
    button_model = YOLO("/Users/dogukanaytekin/PycharmProjects/AtmApp/models/bestLR.pt")
    fingertip_model = YOLO("/Users/dogukanaytekin/PycharmProjects/AtmApp/models/finger_detector.pt")
    WA_model = YOLO("/Users/dogukanaytekin/PycharmProjects/AtmApp/models/WA_model.pt")
    """

    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

    @app.route('/ATMpredict', methods=['POST'])
    def ATMpredict():
        """
         http://127.0.0.1:3000/predict adresine {

          "image_bytes": [255, 216, 255, ...]
         }
         şeklinde resmin post isteği olarak yollanması gerkeiyor dönüt olarak bir json gelicek içindeki result parametresi
         sonucu içeriyor olacak
        """

        data = request.get_json()
        if 'image_bytes' not in data:
            return jsonify({'error': 'No image bytes provided'}), 400

        file_bytes = np.frombuffer(bytearray(data['image_bytes']), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        result = produce_output(image, button_model, fingertip_model)

        return jsonify({'result': result})

    @app.route('/WApredict', methods=['POST'])
    def WApredict():

        data = request.get_json()
        if 'image_bytes' not in data:
            return jsonify({'error': 'No image bytes provided'}), 400

        file_bytes = np.frombuffer(bytearray(data['image_bytes']), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        obstacle_detection_object = ObstacleDetection()
        message = obstacle_detection_object.produce_output(image, WA_model)

        return jsonify({'result': message})

    return app


app = create_app()

if __name__ == '__main__':
    app.run(port=3000, debug=True)
