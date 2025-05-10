import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load model
model = load_model("drone_classifier_model.h5")

# Class labels
labels = ["background", "drone", "helicopter"]

def predict(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=5.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_input = np.expand_dims(mfcc_mean, axis=0)

        prediction = model.predict(mfcc_input)
        label_idx = np.argmax(prediction)
        label = labels[label_idx]

        print(f"\n🔊 Prediction: {label.upper()} (confidence: {prediction[0][label_idx]:.2f})")

    except Exception as e:
        print(f"❌ Error: {e}")

# Command line usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <file.wav>")
    else:
        predict(sys.argv[1])
