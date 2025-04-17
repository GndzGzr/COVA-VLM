import numpy as np
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
import pytesseract
import torch
from app.Obstacle_Detection.Obstacle_Detection import ObstacleDetection
from app.atm import produce_output

button_model = None
fingertip_model = None
WA_model = None
def create_app():
    app = Flask(__name__)

    global button_model, fingertip_model , WA_model

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name='gpu'
    else:
        device = torch.device('cpu')
        device_name = 'cpu'


    button_model = YOLO("/code/bestLR.pt").to(device)
    fingertip_model = YOLO("/code/finger_detector.pt").to(device)
    WA_model = YOLO("/code/WA_model.pt").to(device)
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    """
    local working settings 
    
    button_model = YOLO("/Users/dogukanaytekin/PycharmProjects/AtmApp/models/bestLR.pt")
    fingertip_model = YOLO("/Users/dogukanaytekin/PycharmProjects/AtmApp/models/finger_detector.pt")
    WA_model = YOLO("/Users/dogukanaytekin/PycharmProjects/AtmApp/models/WA_model.pt")
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    """

    # warming up while app is initalized (caching models)
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

    button_model(dummy_image)
    fingertip_model(dummy_image)
    WA_model(dummy_image)
    @app.route('/')
    def home():
        return "Sistem başarıyla başlatıldı." , 200
    @app.route('/ATMpredict', methods=['POST'])
    def ATMpredict():

        data = request.get_json()
        if 'image_bytes' not in data:
            return jsonify({'error': 'No image bytes provided'}), 400

        file_bytes = np.frombuffer(bytearray(data['image_bytes']), np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        result = produce_output(image, button_model, fingertip_model)

        return jsonify({'result': result, 'device': device_name})

    @app.route('/WApredict', methods=['POST'])
    def WApredict():

        data = request.get_json()
        if 'image_bytes' not in data:
            return jsonify({'error': 'No image bytes provided'}), 400

        file_bytes = np.frombuffer(bytearray(data['image_bytes']), np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        obstacle_detection_object = ObstacleDetection()
        message = obstacle_detection_object.produce_output(image, WA_model)

        return jsonify({'result': message , 'device':device_name})

    return app


app = create_app()

if __name__ == '__main__':
    app.run(port=8000, debug=True)
