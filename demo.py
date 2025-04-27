import requests
import argparse
import base64
import json
import cv2
import os
from PIL import Image

def read_image(image_path):
    """Read image from path and convert to bytes for API transmission"""
    with open(image_path, 'rb') as file:
        return bytearray(file.read())

def display_image(image_path, title="Image"):
    """Display the image using OpenCV"""
    image = cv2.imread(image_path)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_atm_predict(api_url, image_path):
    """Test the ATMpredict API endpoint"""
    print("\n---- Testing ATM Prediction API ----")
    
    image_bytes = read_image(image_path)
    
    payload = {
        "image_bytes": list(image_bytes)
    }
    
    try:
        response = requests.post(f"{api_url}/ATMpredict", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"API Response: {result}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error calling ATM Predict API: {e}")
        return None

def test_vlm_predict(api_url, image_path, question):
    """Test the VLMpredict API endpoint"""
    print("\n---- Testing Vision Language Model API ----")
    
    image_bytes = read_image(image_path)
    
    payload = {
        "image_bytes": list(image_bytes),
        "question": question
    }
    
    try:
        response = requests.post(f"{api_url}/VLMpredict", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"Question: {question}")
        print(f"API Response: {result}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error calling VLM Predict API: {e}")
        return None

def test_wa_predict(api_url, image_path, question="What should I do next?"):
    """Test the WApredict API endpoint for walking assistance"""
    print("\n---- Testing Walking Assistance API ----")
    
    image_bytes = read_image(image_path)
    
    payload = {
        "image_bytes": list(image_bytes),
        "question": question
    }
    
    try:
        response = requests.post(f"{api_url}/WApredict", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"Question: {question}")
        print(f"API Response: {result}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error calling Walking Assistance API: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Visual Assistant API Testing Demo")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000", 
                        help="Base URL for the API")
    parser.add_argument("--mode", type=str, choices=["atm", "vlm", "walking", "all"], default="all", 
                        help="Which API to test")
    parser.add_argument("--image_path", type=str, required=True, 
                        help="Path to the image file")
    parser.add_argument("--question", type=str, default="What can you see in this image?", 
                        help="Question for VLM or Walking Assistance")
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return
    
    # Display the image
    display_image(args.image_path)
    
    # Test the appropriate API(s)
    if args.mode == "all" or args.mode == "atm":
        test_atm_predict(args.api_url, args.image_path)
    
    if args.mode == "all" or args.mode == "vlm":
        test_vlm_predict(args.api_url, args.image_path, args.question)
    
    if args.mode == "all" or args.mode == "walking":
        test_wa_predict(args.api_url, args.image_path, args.question)

if __name__ == "__main__":
    main() 