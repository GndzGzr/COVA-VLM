# Visual Assistant API Testing Demos

This repository contains two demo applications for testing the Visual Assistant APIs:

1. **Command-line Demo**: A simple Python script to test APIs from the command line
2. **Web-based Demo**: A Flask web application with an interactive UI for testing the APIs

Both demos support testing all three API endpoints:
- ATM Prediction (`/ATMpredict`)
- Vision Language Model (`/VLMpredict`)
- Walking Assistance (`/WApredict`)

## Getting Started

### Prerequisites

- Python 3.6+
- OpenCV
- Requests
- Flask (for web demo)

You can install the required packages with:

```bash
pip install opencv-python requests flask pillow
```

### Running the Visual Assistant API Server

Before running the demos, you need to have the Visual Assistant API server running. Follow these steps:

1. Start the server using Docker (CPU version):
   ```bash
   docker build --platform linux/amd64 -f Dockerfile.cpu -t visualassistantcpu .   
   docker run -p 8000:8000 --name visualassistancpucontainer visualassistantcpu
   ```

   Or GPU version:
   ```bash
   docker build --platform linux/amd64 -f Dockerfile.gpu -t atm-image .
   docker run --gpus all -d --name atm-container -p 8000:8000 atm-image
   ```

2. Make sure the server is running by opening:
   ```
   http://localhost:8000/
   ```

## Command-line Demo

The command-line demo allows you to test the APIs through simple terminal commands.

### Usage

```bash
python demo.py --image_path <path_to_image> [--api_url <api_url>] [--mode <mode>] [--question <question>]
```

Parameters:
- `--image_path`: (Required) Path to the image file to test
- `--api_url`: (Optional) Base URL for the API (default: http://localhost:8000)
- `--mode`: (Optional) Which API to test - choices: "atm", "vlm", "walking", "all" (default: "all")
- `--question`: (Optional) Question for VLM or Walking Assistance (default: "What can you see in this image?")

For more detailed information, see [demo_README.md](demo_README.md).

## Web-based Demo

The web-based demo provides an interactive interface to test the APIs.

### Usage

1. Run the web demo:
   ```bash
   python web_demo.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Use the web interface to:
   - Enter the API URL
   - Upload an image
   - Select which API to test
   - Enter a question (for VLM and Walking Assistance modes)
   - Submit and view results

For more detailed information, see [web_demo_README.md](web_demo_README.md).

## API Endpoints

### ATM Prediction
Tests the ATM interface interaction detection. Sends an image and returns the detected ATM interface element.

### Vision Language Model
Tests the general vision-language model. Sends an image and a question, and returns the AI's response to the question about the image.

### Walking Assistance
Tests the walking assistance functionality. Sends an image and a question, and returns navigation instructions based on detected obstacles.

## Demo Screenshots

(Add screenshots of the web demo here)

## License

(Add license information here)

## Acknowledgments

(Add acknowledgments here) 