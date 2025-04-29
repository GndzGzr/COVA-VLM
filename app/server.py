import numpy as np
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
import pytesseract
import torch
import os
import time
import gc  # Add garbage collector
from .VLM.vlm_model import VisionLanguageModel
from .Obstacle_Detection.Obstacle_Detection import ObstacleDetection
from .atm import produce_output

# Global model instances
button_model = None
fingertip_model = None
WA_model = None
VLM_model = None
obstacle_detection_object = None  # Initialize obstacle detection once

def initialize_models(device):
    """Initialize all models on the specified device"""
    global button_model, fingertip_model, WA_model, VLM_model, obstacle_detection_object
    
    print(f"Initializing models on {device}...")
    start_time = time.time()
    
    # Set model paths based on environment
    model_base_path = "/code"
    if not os.path.exists(f"{model_base_path}/bestLR.pt"):
        # Use local paths for development
        model_base_path = "/Users/dogukanaytekin/PycharmProjects/AtmApp/models"
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    else:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    
    # Limit CUDA memory usage
    if device == torch.device('cuda'):
        # Reserve only a portion of GPU memory
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of available GPU memory
        torch.cuda.empty_cache()
    
    # Load YOLO models in parallel if on GPU
    button_model = YOLO(f"{model_base_path}/bestLR.pt").to(device)
    fingertip_model = YOLO(f"{model_base_path}/finger_detector.pt").to(device)
    WA_model = YOLO(f"{model_base_path}/WA_model.pt").to(device)
    
    # Initialize obstacle detection
    obstacle_detection_object = ObstacleDetection()
    
    # Initialize VLM model - force CPU for large model if memory issues persist
    use_device_for_vlm = 'cpu'  # Always use CPU for VLM to avoid OOM issues
    VLM_model = VisionLanguageModel(device=use_device_for_vlm)
    
    # Warm up YOLO models with a single dummy image
    print("Warming up YOLO models...")
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run models on dummy image to cache and optimize
    with torch.no_grad():
        button_model(dummy_image)
        fingertip_model(dummy_image)
        WA_model(dummy_image)
    
    # Force garbage collection after initialization
    gc.collect()
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
        
    print(f"All models initialized in {time.time() - start_time:.2f} seconds")

def create_app():
    app = Flask(__name__)

    # Determine device for model execution
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = 'gpu'
        # Set CUDA optimization options
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Set conservative memory usage
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of GPU memory
    else:
        device = torch.device('cpu')
        device_name = 'cpu'
        # Set number of threads for CPU
        torch.set_num_threads(min(os.cpu_count(), 4))

    # Initialize models
    initialize_models(device)

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

        try:
            result = produce_output(image, button_model, fingertip_model)
            # Clean up after processing
            gc.collect()
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()
            return jsonify({'result': result, 'device': device_name})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/VLMpredict', methods=['POST'])
    def VLMpredict():
        data = request.get_json()
        if 'image_bytes' not in data or 'question' not in data:
            return jsonify({'error': 'Image bytes or question not provided'}), 400

        file_bytes = np.frombuffer(bytearray(data['image_bytes']), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        question = data['question']

        try:
            # Use the VLM model in chat mode
            result = VLM_model(image, question, mode="chat")
            print("result: ", result)
            # Clean up after processing
            gc.collect()
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()
            return jsonify({'result': result, 'device': device_name})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/WApredict', methods=['POST'])
    def WApredict():
        data = request.get_json()
        if 'image_bytes' not in data:
            return jsonify({'error': 'No image bytes provided'}), 400

        file_bytes = np.frombuffer(bytearray(data['image_bytes']), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        try:
            # Get detected objects from ObstacleDetection without using VLM
            # Using global obstacle_detection_object to avoid recreating it on each request
            objects_list, detections = obstacle_detection_object.produce_output(image, WA_model)
            
            # Clean up after processing
            gc.collect()
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()
                
            # Return only the objects list without VLM processing
            return jsonify({'objects': objects_list, 'device': device_name, 'detections': detections})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(port=8000, debug=True)
