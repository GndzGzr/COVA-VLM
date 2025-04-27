# Visual Assistant API Demo

This demo application provides an easy way to test the Visual Assistant API endpoints.

## Requirements

- Python 3.6+
- OpenCV
- requests
- PIL (Pillow)

You can install the required packages using:

```bash
pip install opencv-python requests pillow
```

## API Endpoints

This demo can test three API endpoints:

1. **ATM Prediction** (`/ATMpredict`): For ATM interface interaction detection.
2. **Vision Language Model** (`/VLMpredict`): For general vision-language queries.
3. **Walking Assistance** (`/WApredict`): For walking assistance with obstacle detection.

## Usage

```bash
python demo.py --image_path <path_to_image> [--api_url <api_url>] [--mode <mode>] [--question <question>]
```

### Parameters

- `--image_path`: (Required) Path to the image file to test
- `--api_url`: (Optional) Base URL for the API (default: http://localhost:8000)
- `--mode`: (Optional) Which API to test - choices: "atm", "vlm", "walking", "all" (default: "all")
- `--question`: (Optional) Question for VLM or Walking Assistance (default: "What can you see in this image?")

### Examples

Test all APIs with a default question:
```bash
python demo.py --image_path sample_image.jpg
```

Test only the VLM API with a custom question:
```bash
python demo.py --image_path sample_image.jpg --mode vlm --question "What objects are visible in this image?"
```

Test the Walking Assistance API with a custom question:
```bash
python demo.py --image_path street_image.jpg --mode walking --question "Is there any obstacle in front of me?"
```

Test with a remote API:
```bash
python demo.py --image_path sample_image.jpg --api_url http://api.example.com
```

## Notes

- The demo will display the input image before making API calls
- The API responses will be printed to the console
- Make sure the Visual Assistant API server is running before testing 