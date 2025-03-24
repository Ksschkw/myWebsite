from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # <-- Add this
import torch
from chat3 import get_fallback_response, initialize_model, process_message
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # <-- Enable CORS for all routes

model_components = initialize_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        if not user_input or len(user_input) > 500:
            return jsonify({'response': 'Please ask a valid question (max 500 characters)'})

        local_response = process_message(user_input, model_components)
        return jsonify({'response': local_response})

    except Exception as e:
        fallback = get_fallback_response(user_input)
        return jsonify({'response': fallback})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=os.getenv('PORT', 5000))
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 Server starting at http://localhost:{port}")
    print(f"✨ Press CTRL+C to stop\n")
    app.run(host='0.0.0.0', port=port, debug=True)  # Add debug=True here