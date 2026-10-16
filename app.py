import os, re, glob, base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from extract_landmarks import RealTimeKeypointExtractor

app = Flask(__name__)
extractor = RealTimeKeypointExtractor()

SKIP = ['Days_and_Time']      # models you don't want to show yet
MODELS = {}
for path in glob.glob('models/*.keras'):
    name = os.path.splitext(os.path.basename(path))[0]
    actions_file = f'models/{name}_actions.npy'
    if name not in SKIP and os.path.exists(actions_file):
        MODELS[name] = (load_model(path), np.load(actions_file))
        print('[INFO] Loaded model:', name)


def clean(label):
    # "Places_18. City" -> "City"
    return re.sub(r'^.*?_\d+\.\s*', '', str(label))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/categories')
def categories():
    return jsonify(sorted(MODELS.keys()))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model, actions = MODELS[data['category']]

    seq = []
    for f in data['frames']:
        img_bytes = base64.b64decode(f.split(',')[1])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        _, keypoints = extractor.process_frame(img)
        seq.append(keypoints)

    # make exactly 30 frames
    while len(seq) < 30:
        seq.append(seq[-1])
    seq = seq[:30]

    hands_seen = sum(1 for k in seq if k[24:].any())   # frames where a hand was found
    if hands_seen < 8:
        return jsonify({'word': None, 'message': 'No hands detected. Show your hands to the camera.'})

    probs = model.predict(np.expand_dims(np.array(seq), 0), verbose=0)[0]
    best = int(np.argmax(probs))
    return jsonify({
        'word': clean(actions[best]),
        'confidence': float(probs[best]),
        'message': ''
    })


if __name__ == '__main__':
    app.run(debug=False)