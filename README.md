# Sign Language Recognition Web App

A web app that recognizes sign language words from a webcam. The user picks a category, signs a word, and the app shows it as text and can speak it aloud.

## How it works
Camera (30 frames) -> MediaPipe Holistic keypoints (150 numbers per frame) -> BiLSTM model -> predicted word -> text and speech

## Tech used
- Python, Flask
- TensorFlow / Keras (BiLSTM model, Adam optimizer)
- MediaPipe Holistic, OpenCV, NumPy, scikit-learn
- HTML, CSS, JavaScript (webcam and text-to-speech in the browser)

## Categories
One model per category: Places, Jobs, Animals, Greetings.

## Run the app
1. Install the libraries:
   pip install -r requirements.txt
2. Start the server:
   python app.py
3. Open http://127.0.0.1:5000 in your browser and allow the camera.

## Project files
- extract_landmarks.py: extracts body and hand keypoints
- process_all_dataset.py: converts videos to keypoints
- train_category.py: trains one model per category
- count.py: counts clips per word
- data_collection.py: records clips from the webcam
- app.py and templates/index.html: the web app
- models/: trained models

## Note
The dataset (Raw_Dataset and MP_Data) is not included because of its size.

## Author
sakshi442
