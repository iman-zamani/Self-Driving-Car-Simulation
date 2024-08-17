# Self-Driving Car Simulation Project

Welcome to the Self-Driving Car Simulation project! This repository showcases a small-scale self-driving car simulation using TensorFlow models designed for various tasks such as road line detection, traffic sign detection, and arrow direction detection. These models can be run using Python, TensorFlow Lite, and OpenCV, and they are designed to work on both live webcam feeds and video files.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup Instructions](#setup-instructions)
4. [Running the Models](#running-the-models)
5. [Using GPU](#using-gpu)
6. [License](#license)

## Video of raod line detecting 

![Output GIF](road_line/output.gif)

## Project Overview

This project is a simulation of a self-driving car on a small scale, like the size of a toy car...


## Project Overview

This project is a simulation of a self-driving car on a small scale, like the size of a toy car...


## Project Overview

This project is a simulation of a self-driving car on a small scale, like the size of a toy car. The project is divided into three main directories, each focusing on a different aspect of autonomous driving:

1. **Road Line Detection:** Detects lines on the road using TensorFlow Lite.
2. **Sign Detection:** Identifies various traffic signs using TensorFlow Lite.
3. **Arrow Direction Detection:** Determines the direction of arrows detected by the sign detection model.

## Directory Structure

The repository is organized into three main directories:

### 1. `road_line/`
- **Purpose:** Detects the lines on the road using TensorFlow Lite models.
- **Files:**
  - `detect.tflite`: TensorFlow Lite model for line detection.
  - `labelmap.txt`: Label map for the model.
  - `webcam.py`: Runs the model on a live webcam feed.
  - `detectionOnVideo.py`: Runs the model on a pre-recorded video file (`video.mp4`) and saves the processed video with detected road lines as `output.mp4`.
  - `video.mp4`: Sample video to test the model.
  - `output.mp4`: The output video generated after running `detectionOnVideo.py` on `video.mp4`.

### 2. `sign_detection/`
- **Purpose:** Detects traffic signs in images or video streams.
- **Files:**
  - `model.tflite`: TensorFlow Lite model for traffic sign detection.
  - `main.py`: Runs the model on a live webcam feed and displays the detected traffic sign.

### 3. `ArrowDirection/`
- **Purpose:** Determines the direction of an arrow detected by the sign detection model.
- **Files:**
  - `detect.tflite`: TensorFlow Lite model for arrow direction detection.
  - `labelmap.txt`: Label map for the model.
  - `main.py`: Runs the model on a live webcam feed and displays the direction of the detected arrow.

## Setup Instructions

To run the models, you'll need to set up a Python environment with the required dependencies.

### 1. Create a Python Virtual Environment
```bash
python3 -m venv self_driving_env
source self_driving_env/bin/activate  # On Windows: self_driving_env\Scripts\activate
```

### 2. Install Required Packages
```bash
pip install tensorflow opencv-python
```

### 3. Verify Installation
To verify that TensorFlow and OpenCV are installed correctly, you can run the following commands in the Python interpreter:
```python
import tensorflow as tf
import cv2
print(tf.__version__)
print(cv2.__version__)
```

## Running the Models

### 1. Road Line Detection

#### a. Running on Webcam
To detect road lines using a live webcam feed, navigate to the `road_line/` directory and run:
```bash
python webcam.py
```
This script captures video from your webcam and uses the TensorFlow Lite model to detect road lines in real-time.

#### b. Running on Video File

To detect road lines on a pre-recorded video file and save the output, run:

```bash
python detectionOnVideo.py
```

This script uses the TensorFlow Lite model to detect road lines in the provided `video.mp4` file. The processed video with the detected road lines will be saved as `output.mp4` in the same directory.

### 2. Sign Detection

Navigate to the `sign_detection/` directory and run:

```bash
python main.py
```

This script uses your webcam to detect traffic signs and displays the type of sign detected. The script is capable of detecting the following traffic signs:

- **Do Not Enter**
- **Directional Arrow**
- **Stop**
- **Dead End**
- **No Sign**

Based on the detected sign, the script will display the corresponding name on the screen.


### 3. Arrow Direction Detection

Navigate to the `ArrowDirection/` directory and run:

```bash
python main.py
```

This script uses your webcam to detect the direction of arrows. It works by identifying two key points on the arrow: the end (tail) and the tip (head). By comparing the positions of these points, the script determines the direction in which the arrow is pointing (left, right, up, or down). The output will display the direction based on this analysis.

## Using GPU

To accelerate the inference of TensorFlow models using a GPU, ensure you have the necessary GPU drivers and CUDA toolkit installed. You can install the GPU-enabled version of TensorFlow with:

```bash
pip install tensorflow-gpu
```

For detailed instructions and system requirements, please visit the [TensorFlow GPU support guide](https://www.tensorflow.org/install/gpu).

Make sure your system meets the requirements for running TensorFlow with GPU support. For most small-scale projects like this, the CPU version should suffice, but for more intensive tasks, GPU acceleration can significantly speed up processing.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
If you encounter any issues or have questions, feel free to open an issue or contribute to the project!
