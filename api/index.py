import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import PIL.Image
import io
import os

app = Flask(__name__)
CORS(app)

# --- Configuration & Model Initialization ---
# This safely reads the API key from Vercel's Environment Variables
API_KEY = os.environ.get('GEMINI_API_KEY')
model = None

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        # This will log an error in your Vercel dashboard if something goes wrong
        print(f"CRITICAL ERROR: Failed to configure or initialize Gemini model. Details: {e}")
else:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not found!")

prompt_text = """
Analyze the animal in this image and identify its specific breed.
Your entire response MUST BE ONLY the breed's name and nothing else.
Focus exclusively on identifying common Indian breeds of cattle and buffaloes.
Examples: Gir Cow, Murrah Buffalo, Sahiwal Cattle.
If not identifiable, respond with 'Breed not identifiable'.
"""

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Server configuration error: Model not initialized.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        img = PIL.Image.open(file.stream)
        response = model.generate_content([prompt_text, img])
        breed = response.text.strip()
        return jsonify({'breed': breed})

    except Exception as e:
        print(f"CRITICAL ERROR: Exception during Gemini API call: {e}")
        return jsonify({'error': 'An error occurred on the server while processing the image.'}), 500

