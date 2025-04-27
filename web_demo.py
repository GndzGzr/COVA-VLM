import os
import requests
import base64
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if API URL is provided
    api_url = request.form.get('api_url', 'http://localhost:8000')
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Get the API mode
    mode = request.form.get('mode', 'all')
    question = request.form.get('question', 'What can you see in this image?')
    
    # Read the file as bytes
    with open(filepath, 'rb') as f:
        image_bytes = bytearray(f.read())
    
    results = {}
    
    # Call appropriate API(s)
    if mode == 'all' or mode == 'atm':
        try:
            payload = {"image_bytes": list(image_bytes)}
            response = requests.post(f"{api_url}/ATMpredict", json=payload)
            if response.status_code == 200:
                results['atm'] = response.json()
            else:
                results['atm'] = {'error': f"API returned status code {response.status_code}"}
        except Exception as e:
            results['atm'] = {'error': str(e)}
    
    if mode == 'all' or mode == 'vlm':
        try:
            payload = {"image_bytes": list(image_bytes), "question": question}
            response = requests.post(f"{api_url}/VLMpredict", json=payload)
            if response.status_code == 200:
                results['vlm'] = response.json()
            else:
                results['vlm'] = {'error': f"API returned status code {response.status_code}"}
        except Exception as e:
            results['vlm'] = {'error': str(e)}
    
    if mode == 'all' or mode == 'walking':
        try:
            payload = {"image_bytes": list(image_bytes), "question": question}
            response = requests.post(f"{api_url}/WApredict", json=payload)
            if response.status_code == 200:
                results['walking'] = response.json()
            else:
                results['walking'] = {'error': f"API returned status code {response.status_code}"}
        except Exception as e:
            results['walking'] = {'error': str(e)}
    
    # Return the filepath (for displaying the image) and the results
    return jsonify({
        'filepath': f"{filename}",
        'results': results
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    # Create static/uploads symlink to uploads folder
    os.makedirs('static', exist_ok=True)
    static_uploads = os.path.join('static', 'uploads')
    
    if not os.path.exists(static_uploads):
        try:
            # Try to create a symlink (works on Unix-like systems)
            os.symlink(os.path.abspath(app.config['UPLOAD_FOLDER']), static_uploads)
        except:
            # If symlink fails (e.g., on Windows), create the directory
            os.makedirs(static_uploads, exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, port=5000) 