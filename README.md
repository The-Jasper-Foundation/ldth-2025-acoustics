# Drone Sound Classifier 🎧🛸

This project was developed for the **London Defence Tech Hackathon 2025**. It uses machine learning to classify audio recordings into three categories:
- 🚁 Helicopter
- 🛸 Drone
- 🌿 Background noise

## 🔍 How it works
A neural network is trained on acoustic data using the `librosa` and `TensorFlow` libraries. The dataset is split into training and validation sets, and performance is tracked over 30 epochs.

## 🧠 Model Accuracy
- ✅ Final Training Accuracy: ~100%
- ✅ Final Validation Accuracy: ~94.8%

## 🚀 How to Use
```bash
uv sync
uv run src/ldth_drone_acoustics/setup/download_raw_data.py
uv run train_model.py


# LDTH 2025 - Helsing Drone Acoustics

> Machine learning on drone acoustics for the London Defense Tech Hackathon, May 2025. Designed and run by Helsing.

## Challenge Prompt

Our AI problem is framed around detecting drones using acoustic data.
Automated detection of threats is essential in facilitating early warning and situational awareness.
Acoustic detection complements other sensor modalities; while radar, optical, and infrared sensors can also be
used for this problem, each has limitations such as weather and obstructions.
Given the low infrastructure costs and ability for rapid deployment, acoustic sensing presents a suitable additional
layer of surveillance for modern defense strategies.

The problem is split into two phases.

Phase 1: 3-class prediction. We provide a small curated dataset of open-source acoustic recordings split into three
categories: background, drone, and helicopter. The challenge is to train a model to separate these three class from
their acoustic signatures.

Phase 2: Enhanced prediction. Creating AI to use in the field is not just about model performance. We also need to
consider aspects such as inference time, edge support, and assurance. To this end, we ask contestants to explore the
ways they can enhance their approach for use in the field. This is intentionally left quite open-ended: we
want you to be creative! However, some suggestions include: analysing the interpretability/explainability of your
model, using as lightweight or as fast a model as possible (while maintaining predictive accuracy!), or creating new
synthetic data to explore what happens with really quiet contacts.

## Data

Sourced from: https://github.com/DroneDetectionThesis/Drone-detection-dataset (audio + video dataset)  
Paper: [A dataset for multi-sensor drone detection](https://www.sciencedirect.com/science/article/pii/S2352340921007976#!)

### Audio Dataset Details

While the GitHub provides both audio and video, we are only interested in the audio data.  
The challenge is to perform three-class classification (background/drone/helicopter) purely from audio.  
Audio is captured from a Boya BY-MM1 mini cardioid directional microphone.  
The provided audio in two channel L/R format, which has been automatically processed from a mono microphone.  
For each 2-channel 10-second file, we convert this into single channel (left or right) non-overlapping 5 second clips.  
This means each individual original file becomes four distinct files in our dataset.

From the paper:  
_The audio part has 90 ten-second files in wav-format with a sampling frequency of 44100 Hz.  
There are 30 files of each of the three output audio classes [background, drone, helicopter].  
The clips are annotated with the filenames themselves, e.g. DRONE_001.wav, HELICOPTER_001.wav, BACKGROUND_001.wav, etc.  
The audio in the dataset is taken from the videos or recorded separately.  
The background sound class contains general background sounds recorded outdoor in the acquisition location and
includes some clips of the sounds from the servos moving the pan/tilt platform where the sensors were mounted._

## Getting Started

This project uses `uv` for dependency management.
See [installing uv](https://docs.astral.sh/uv/getting-started/installation/) and then run `uv sync` to setup your venv.

To access the dataset, we have provided the script [src/ldth_drone_acoustics/setup/download_raw_data.py](src/ldth_drone_acoustics/setup/download_raw_data.py).  
Run `uv run src/ldth_drone_acoustics/setup/download_raw_data.py` to download the data.

We suggest forking this repo so you can use version control in your own space.

Happy hacking!
