# Visual Assistant Web Demo

This is a web-based demo application for testing the Visual Assistant API endpoints with a user-friendly interface.

## Requirements

- Python 3.6+
- Flask
- requests
- Werkzeug

You can install the required packages using:

```bash
pip install flask requests werkzeug
```

## Features

- Web interface for uploading images and testing API endpoints
- Support for all three Visual Assistant API endpoints:
  1. ATM Prediction
  2. Vision Language Model
  3. Walking Assistance
- Configurable API URL
- Image preview
- Formatted JSON response display

## Directory Structure

```
/
├── web_demo.py                # Main Flask application
├── templates/
│   └── index.html            # HTML template for the web interface
├── static/
│   └── uploads/              # Symlink to uploads directory (created automatically)
└── uploads/                  # Directory for storing uploaded images
```

## Usage

1. Start the Visual Assistant API server (if running locally):
   ```bash
   # Start the backend API (follow instructions in the main README)
   ```

2. Run the web demo:
   ```bash
   python web_demo.py
   ```

3. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

4. Use the web interface to:
   - Enter the API URL (default: http://localhost:8000)
   - Upload an image
   - Select which API to test
   - Enter a question (for VLM and Walking Assistance modes)
   - Submit the form and view the results

## Notes

- The web demo will create an 'uploads' directory to store uploaded images
- The maximum file size for uploads is 16MB
- If testing with a remote API, make sure to enter the correct API URL
- The server runs in debug mode by default (not recommended for production) 