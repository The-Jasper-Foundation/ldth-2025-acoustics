import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Define data path and class labels
DATA_DIR = "data/raw/train"
CLASSES = ["background", "drone", "helicopter"]
SAMPLES, LABELS = [], []

# Load and process audio files
for label, class_name in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, class_name)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            audio_path = os.path.join(folder, file)
            audio, sr = librosa.load(audio_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            SAMPLES.append(mfcc_mean)
            LABELS.append(label)

# Prepare data
X = np.array(SAMPLES)
y = to_categorical(LABELS)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, input_shape=(13,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# Save model
model.save("drone_classifier_model.h5")
