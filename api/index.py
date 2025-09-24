import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import PIL.Image
import io
import os

app = Flask(__name__)
CORS(app)

# --- Configuration ---
# SECURE: Reads the API key from Vercel's Environment Variables
API_KEY = os.environ.get('GEMINI_API_KEY')

# --- The AI Model and Prompt ---
prompt_text = """
Analyze the animal in this image and identify its specific breed.
Your entire response MUST BE ONLY the breed's name and nothing else.
Focus exclusively on identifying common Indian breeds of cattle and buffaloes.

Here are examples of valid, expected responses:
- Gir Cow
- Murrah Buffalo
- Sahiwal Cattle

If you cannot determine the breed with high confidence, your response should be exactly: 'Breed not identifiable'. Do not add any other words.
"""

# Initialize model variable
model = None
try:
    if API_KEY:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest') 
        print("INFO: Gemini API configured successfully.")
    else:
        print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not configure Gemini API. Details: {e}")

@app.route('/api/predict', methods=['POST'])
def predict():
    # The rest of your predict function remains the same...
    # (Code from your original app.py)
    if not model:
        return jsonify({'error': 'Gemini API key is not configured correctly on the server.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    try:
        img = PIL.Image.open(file.stream)
        response = model.generate_content([prompt_text, img], request_options={"timeout": 100})
        breed = response.text.strip()
        return jsonify({'breed': breed})
    except Exception as e:
        error_message = f"Gemini API Error: {type(e).__name__}."
        return jsonify({'error': error_message}), 500

# Note: The app.run() line is removed for Vercel deployment
