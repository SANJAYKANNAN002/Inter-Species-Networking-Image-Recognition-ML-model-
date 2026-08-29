# Inter-Species-Networking-Image-Recognition-ML-model-
AI-powered wildlife monitoring and poaching detection system using YOLO11. It detects 72 wildlife, human, and weapon classes from images, identifies potential poaching activity, and provides real-time threat assessments through an interactive Streamlit web application.


# Wildlife Monitoring & Poaching Detection System

> AI-powered wildlife monitoring and potential poaching activity detection using YOLO11 and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO11-Ultralytics-purple.svg)](https://docs.ultralytics.com/models/yolo11/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-See%20LICENSE-lightgrey.svg)](LICENSE)

---

## Overview

The **Wildlife Monitoring & Poaching Detection System** is a computer-vision application designed to assist in the identification of wildlife, human activity, and potential poaching-related objects from images.

The system combines a custom-trained **YOLO11 object detection model** with a **Streamlit-based web interface**. Users can upload an image through the application, after which the trained model performs object detection and returns:

- Detected object classes
- Bounding boxes
- Confidence scores
- Number of detected instances
- A rule-based threat assessment based on detected classes

The model was trained on a custom wildlife-monitoring dataset containing **72 object classes**, including wildlife species, vehicles, humans, and potential weapons or poaching-related equipment.

The application is intended as a **decision-support and monitoring prototype**. It is not designed to replace trained wildlife personnel, law-enforcement officers, or professional surveillance systems.

---

# Key Features

## 1. Multi-Class Object Detection

The system is trained to detect 72 classes spanning several categories:

### Wildlife

Examples include:

- Antelope
- Bear
- Cheetah
- Chimpanzee
- Deer
- Dog
- Elephant
- Fox
- Giraffe
- Gorilla
- Horse
- Kangaroo
- Leopard
- Lion
- Panda
- Tiger
- Wolf
- Zebra
- and other wildlife classes

### Human Activity

- Hunter

### Weapons / Potential Poaching Equipment

- Knife
- Pistol
- Rifle
- X-Bow

### Other Objects

- Car
- Truck
- Bike
- Jeep
- Helicopter
- Binocular
- Rope
- Van
- and other classes included in the dataset

The complete class mapping is maintained according to the original dataset configuration.

---

## 2. YOLO11-Based Detection

The detection pipeline uses **YOLO11n**, a lightweight YOLO11 model suitable for fast object detection.

The model performs:

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
YOLO11 Object Detection
     │
     ├── Class Identification
     ├── Bounding Box Localization
     └── Confidence Estimation
     │
     ▼
Threat Assessment
     │
     ▼
Streamlit Visualization



*System Architecture*
                         ┌──────────────────────┐
                         │      User Input      │
                         │      Image Upload    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         │      app/app.py      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ WildlifeDetector     │
                         │  app/detector.py     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      YOLO11n         │
                         │    best_fixed.pt     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Object Detections    │
                         │                      │
                         │ • Class              │
                         │ • Bounding Box       │
                         │ • Confidence         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Threat Assessment    │
                         │   app/utils.py       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Results & Visualization│
                         │      Streamlit       │
                         └──────────────────────┘



*TECH STACK USED*
  | Component               | Technology                         |
| ----------------------- | ---------------------------------- |
| Programming Language    | Python 3.11                        |
| Deep Learning Framework | PyTorch                            |
| Object Detection        | Ultralytics YOLO11                 |
| Model Variant           | YOLO11n                            |
| Web Framework           | Streamlit                          |
| Image Processing        | OpenCV / Pillow                    |
| Numerical Computing     | NumPy                              |
| Local GPU               | NVIDIA GeForce RTX 4050 Laptop GPU |
| Local CUDA              | CUDA 12.6                          |
| Model Format            | PyTorch `.pt`                      |
| Deployment Target       | Streamlit Community Cloud          |


*Dataset*

The project uses a custom wildlife-monitoring dataset containing:

72 object classes
Training images
Validation images
Test images
YOLO-format bounding-box annotations

The original dataset class mapping is preserved in the project workflow to ensure that numerical class IDs correspond to their correct semantic labels.

*Dataset Splits*

The dataset was organized into:
dataset/
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/

Training images:    4,164
Validation images:    901


*Class Mapping*

The dataset contains 72 classes indexed from 0 to 71.

The official mapping used by the project is:
| ID | Class        |
| -: | ------------ |
|  0 | Antelope     |
|  1 | Badger       |
|  2 | Bat          |
|  3 | Bear         |
|  4 | Bike         |
|  5 | Binocular    |
|  6 | Bison        |
|  7 | Boar         |
|  8 | Car          |
|  9 | Cheetah      |
| 10 | Chimpanzee   |
| 11 | Coyote       |
| 12 | Deer         |
| 13 | Dog          |
| 14 | Donkey       |
| 15 | Duck         |
| 16 | Eagle        |
| 17 | Elephant     |
| 18 | Flamingo     |
| 19 | Fox          |
| 20 | Giraffe      |
| 21 | Goat         |
| 22 | Goose        |
| 23 | Gorilla      |
| 24 | Hare         |
| 25 | Hedgehog     |
| 26 | Helicopter   |
| 27 | Hippopotamus |
| 28 | Hornbill     |
| 29 | Horse        |
| 30 | Humming Bird |
| 31 | Hunter       |
| 32 | Hyena        |
| 33 | Jeep         |
| 34 | Kangaroo     |
| 35 | Knife        |
| 36 | Koala        |
| 37 | Leopard      |
| 38 | Lion         |
| 39 | Lizard       |
| 40 | Mouse        |
| 41 | Okapi        |
| 42 | Orangutan    |
| 43 | Otter        |
| 44 | Owl          |
| 45 | Ox           |
| 46 | Panda        |
| 47 | Parrot       |
| 48 | Pig          |
| 49 | Pigeon       |
| 50 | Pistol       |
| 51 | Porcupine    |
| 52 | Possum       |
| 53 | Raccoon      |
| 54 | Reindeer     |
| 55 | Rifle        |
| 56 | Rinoceros    |
| 57 | Rope         |
| 58 | Sandpiper    |
| 59 | Sheep        |
| 60 | Snake        |
| 61 | Sparrow      |
| 62 | Squirrel     |
| 63 | Tiger        |
| 64 | Truck        |
| 65 | Turkey       |
| 66 | Van          |
| 67 | Wolf         |
| 68 | Wombat       |
| 69 | Woodpecker   |
| 70 | X-Bow        |
| 71 | Zebra        |


*Model Training*

The project uses a pretrained YOLO11n model as the starting point and fine-tunes it on the custom dataset.

The training environment used during development was:
Python:      3.11.9
PyTorch:     2.13.0+cu126
CUDA:        12.6
GPU:         NVIDIA GeForce RTX 4050 Laptop GPU
VRAM:        ~6 GB
Image Size:  640 × 640
Batch Size:  4


The model was trained using the Ultralytics YOLO training pipeline.

Example training command:
yolo detect train \
    model=yolo11n.pt \
    data=dataset/data.yaml \
    epochs=40 \
    imgsz=640 \
    batch=4 \
    device=0
Ultralytics supports YOLO11 training through both Python and CLI workflows.



*Model Evaluation*

The best checkpoint obtained during training was evaluated on the validation set.

Validation Results
Images:       901
Instances:    1,075

Precision:    0.747
Recall:       0.712
mAP@50:       0.766
mAP@50-95:    0.572

These metrics indicate that the model provides useful object-detection performance on the validation data, while also showing that performance varies between classes.

For example, some classes achieved substantially stronger results than others due to differences in image count, appearance, object scale, and visual similarity.

Therefore, the system should be considered a prototype decision-support tool rather than a fully autonomous surveillance or law-enforcement system.



*Model Artifact*

The application uses:

model/best_fixed.pt

The model is approximately 5.3 MB.

The deployed application uses this checkpoint rather than the original best.pt because the final checkpoint contains the corrected class-name metadata corresponding to the original dataset mapping.

The original model is intentionally not included in the deployment package.



*Inference Pipeline*

The inference process is implemented in:

app/detector.py

The detector:

Loads best_fixed.pt
Receives an uploaded image
Runs YOLO inference
Applies a configurable confidence threshold
Returns Ultralytics detection results
Passes the detections to the application layer
Displays annotated results in Streamlit

Ultralytics' prediction API accepts image sources and returns detection results containing predicted classes and object locations.



Threat Assessment

Object detection and threat assessment are treated as separate layers.

The YOLO model is responsible for answering:

"What objects are present in the image?"

The application logic then interprets those detections to estimate:

"Does this scene contain indicators of potential poaching activity?"

Potentially high-risk classes include:

Hunter
Rifle
Pistol
Knife
X-Bow

Wildlife classes are not inherently classified as threats.

The application can therefore distinguish between scenarios such as:

Wildlife only
        ↓
Low / No immediate poaching indicator

and:

Hunter + Weapon
        ↓
Potential poaching activity
        ↓
High-risk assessment

This separation between perception and application-level decision logic makes the system easier to modify and evaluate.

Web Application

The frontend is implemented using Streamlit.

The application provides:

Image upload
Image preview
Object detection
Bounding-box visualization
Class labels
Confidence scores
Detection summaries
Threat assessment
Forest-inspired visual design

The application is structured so that the user does not need to interact directly with the YOLO model or Python inference code.

Project Structure
wildlife-monitoring-poaching-detection/
│
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── detector.py
│   └── utils.py
│
├── model/
│   └── best_fixed.pt
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
app/app.py

Main Streamlit application.

Responsible for:

User interface
Image upload
Result display
Application flow
app/detector.py

Model inference layer.

Responsible for:

Loading the trained YOLO model
Running inference
Returning detection results
app/utils.py

Utility and application-level helper functions.

model/best_fixed.pt

Final trained YOLO11n checkpoint used for inference.

requirements.txt

Python dependencies required by the application.

Streamlit Community Cloud uses dependency files such as requirements.txt to construct the application's Python environment.

Installation
Prerequisites

Recommended local environment:

Python 3.11
Git

For GPU-accelerated local inference/training:

NVIDIA GPU
Compatible NVIDIA driver
CUDA-compatible PyTorch installation

The application can also run inference on CPU.

1. Clone the Repository
git clone https://github.com/<YOUR_USERNAME>/wildlife-monitoring-poaching-detection.git
cd wildlife-monitoring-poaching-detection
2. Create a Virtual Environment
Windows
python -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Running the Application Locally

From the repository root:

streamlit run app/app.py

The application will provide a local URL, typically:

http://localhost:8501

Open the URL in a browser and upload an image for analysis.

Running YOLO Inference Directly

The trained model can also be used without Streamlit.

Example:

yolo detect predict \
    model="./model/best_fixed.pt" \
    source="./path/to/image.jpg" \
    conf=0.15 \
    save=True

The Ultralytics CLI supports prediction directly from trained model checkpoints.

Python Inference Example
from ultralytics import YOLO

model = YOLO("model/best_fixed.pt")

results = model.predict(
    source="image.jpg",
    conf=0.25,
    device="cpu"
)

for result in results:
    print(result.boxes)


*Deployment*

The application is designed to be deployable using Streamlit Community Cloud.

The repository contains:

app/app.py
requirements.txt
model/best_fixed.pt

which are sufficient for the application to load the model and perform inference.

Streamlit Community Cloud runs the application from the repository root and supports entrypoint files located in subdirectories, making app/app.py a valid application entrypoint.



*Deployment configuration*

Repository:

<GitHub username>/wildlife-monitoring-poaching-detection

Branch:

main

Application entrypoint:

app/app.py

The deployment environment uses CPU inference rather than assuming access to the developer's local NVIDIA GPU.

Testing Strategy

Testing was performed at multiple levels.

1. Dataset-Level Testing

The model was evaluated using the validation dataset and standard object-detection metrics.

2. Class-Mapping Verification

The original dataset's class-ID mapping was verified to ensure that numerical labels corresponded to the correct semantic classes.

Examples:

31 → Hunter
52 → Possum
55 → Rifle
63 → Tiger
3. Individual Image Testing

Representative test images were used to verify specific classes.

Examples included:

Possum → Possum
Tiger → Tiger
Hunter → Hunter
Hunter + weapon → Hunter + Rifle
Binocular → Binocular
4. Application-Level Testing

The same model checkpoint was tested through the Streamlit interface to ensure that the web application displayed the corrected class names and detection results.

Performance Considerations

YOLO11n was selected because the project requires a balance between:

Detection accuracy
Model size
Inference speed
Hardware requirements
Deployment feasibility

The local development environment used an NVIDIA RTX 4050 Laptop GPU.

Example local inference performance during testing was approximately:

~10–21 ms model inference time per image

depending on the input and execution state.

Cloud deployment uses CPU inference and is therefore expected to be slower than local GPU inference.



*Limitations*

The current system has several important limitations.

1. Dataset Dependency

Performance depends heavily on the quality, diversity, and distribution of the training dataset.

2. Class Imbalance

Not every class has the same number or diversity of examples, which can result in uneven per-class performance.

3. Visual Similarity

Wildlife species with similar visual characteristics may be difficult to distinguish.

4. Environmental Conditions

Performance may decrease under:

Poor lighting
Heavy vegetation
Occlusion
Motion blur
Unusual camera angles
Low-resolution imagery
Partial object visibility
5. False Positives and False Negatives

Object detectors can produce:

False detections
Missed detections
Incorrect class predictions
Incorrect bounding-box localization
6. Threat Assessment Is Rule-Based

The threat assessment layer is not itself a learned behavioral model.

Detecting a person, weapon, or wildlife object does not prove that illegal activity is taking place.

The output should therefore be interpreted as an indicator for further review, not as definitive evidence of poaching.



*Future Improvements*

Several improvements can be considered for future versions.

Model Improvements
Increase dataset size
Add more diverse environmental conditions
Improve class balance
Add hard-negative examples
Perform systematic hyperparameter tuning
Evaluate larger YOLO model variants
Introduce stronger augmentation strategies
Perform cross-validation where appropriate
Detection Improvements
Video-stream inference
Object tracking
Temporal activity analysis
Multi-frame threat assessment
Camera-specific calibration
Night-time detection support
Application Improvements
Detection history
Event logging
Timestamped alerts
Location metadata
Dashboard analytics
Detection confidence filtering
Exportable incident reports
Deployment Improvements
GPU-enabled inference infrastructure
Containerized deployment
API-based inference service
Authentication
Monitoring and logging
Scalable backend architecture
Responsible Use

This project is intended for research, education, prototyping, and decision-support purposes.

The system should not be used as the sole basis for:

Law-enforcement action
Arrest or detention
Wildlife intervention
Identification of individuals
Automated accusations of criminal activity

Model predictions should be reviewed by qualified personnel before any operational decision is made.


**Reproducibility**

To reproduce the development environment:

Python 3.11
PyTorch 2.13
CUDA 12.6
Ultralytics YOLO11
Streamlit

The project stores the final inference checkpoint in:

model/best_fixed.pt

The training dataset itself is not included in the deployment repository.

Development Workflow

The development process followed the following pipeline:

Dataset Acquisition
        │
        ▼
Dataset Organization
        │
        ▼
YOLO Label Verification
        │
        ▼
Class Mapping Verification
        │
        ▼
YOLO11n Fine-Tuning
        │
        ▼
Validation
        │
        ▼
Model Checkpoint Selection
        │
        ▼
Class Metadata Correction
        │
        ▼
Independent Test Images
        │
        ▼
Streamlit Integration
        │
        ▼
Application Testing
        │
        ▼
Deployment
License

This project is distributed under the license included in:

*LICENSE*

The dataset and third-party components may have their own licensing requirements.

The dataset used during development was obtained from Roboflow Universe and should be used in accordance with its applicable license and attribution requirements.

Ultralytics YOLO11 is distributed under the licensing terms specified by Ultralytics. Review the applicable Ultralytics license before using the project commercially.

*Acknowledgements*

This project makes use of:

Ultralytics YOLO11 for object detection
PyTorch for deep-learning inference and training
Streamlit for the interactive web application
Roboflow Universe for the wildlife-monitoring dataset used during development
References
Ultralytics YOLO11 Documentation
https://docs.ultralytics.com/models/yolo11/
Ultralytics Python Usage
https://docs.ultralytics.com/usage/python
Streamlit Documentation
https://docs.streamlit.io/
Streamlit Community Cloud Deployment
https://docs.streamlit.io/deploy/streamlit-community-cloud
Project Status

Status: Prototype / Academic Project

The current implementation successfully demonstrates:

Custom multi-class object detection
Wildlife identification
Detection of human/poaching-related classes
Bounding-box visualization
Confidence scoring
Rule-based threat assessment
Interactive Streamlit interface
Local GPU inference
CPU-compatible deployment architecture

Future development can extend the system toward video surveillance, tracking, alerting, event logging, and scalable deployment.


### One change I'd make before committing this README

Your current `README.md` is already staged/committed in Git. After replacing it with this version, you'll need to commit the updated README:

```powershell
git add README.md
git commit -m "Improve project documentation"
