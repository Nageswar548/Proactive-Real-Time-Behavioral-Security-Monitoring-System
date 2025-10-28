# 🌟 Proactive Real-Time Behavioral Security Monitoring  System

### Project Description

This system is an advanced Computer Vision solution designed for **real-time human action and behavioral classification** in security environments. It moves beyond simple object detection by performing sophisticated **Pose Estimation** and classifying actions (e.g., 'Using_Phone', 'Sitting') with high accuracy.

The project demonstrates successful integration and deployment of multiple complex Python libraries for real-world inference.

### 🎯 Key Performance & Results

| Metric | Detail | Evidence |
| :--- | :--- | :--- |
| **Model Accuracy** | Custom-trained Scikit-learn model achieved **99.06%** test accuracy. |  |
| **Real-Time Output**| Successfully detects human pose (key points) and outputs the classification score. |  |
| **Core Function** | Identifies behavior for proactive monitoring against pre-defined actions. |

### 🛠️ Technology Stack
## 💻 Tech Stack & Tools

![Python 3.x](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white) 
![OpenCV](https://img.shields.io/badge/Library-OpenCV-brightgreen?style=flat&logo=opencv) 
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange?style=flat&logo=scikit-learn) 
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-red?style=flat&logo=yolo) 
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat&logo=pandas)

| Library | Function | Technical Skill Demonstrated |
| :--- | :--- | :--- |
| **Ultralytics (YOLO)** | Core framework for efficient object and pose detection. | Deep Learning CV Framework Deployment |
| **MediaPipe** | Robust library for detailed human body and face landmark estimation. | Multi-Library Integration |
| **Scikit-learn** | Final machine learning classifier for high-accuracy action prediction. | ML Model Training & Deployment |
| **OpenCV, Pandas** | Video stream processing and structured data management. | Data Pipeline Efficiency |

### 🛑 The Technical Journey (The Real Story)

This project’s successful deployment required aggressive and **proactive debugging** to overcome major setup hurdles:

* **Path Conflict Resolution:** Resolved persistent **`ModuleNotFoundError`** errors (affecting `ultralytics`, `mediapipe`, `sklearn`, and `pandas`) that were caused by deep-rooted conflicts between the system's global Python path and the isolated virtual environment.
* **Environment Isolation:** Successfully created a fully isolated virtual environment (`venv_cv`) using explicit path calls and forced clean installation to ensure all packages resolved to the correct local directory.

### ▶️ Setup & Execution

To run the system, follow these simple steps using the fixed environment:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Nageswar548/Proactive-Real-Time-Behavioral-Security-Monitoring-System]
    cd CV_Security_Monitor
    ```

2.  **Create & Activate Environment:**
    ```bash
    python -m venv venv_cv
    .\venv_cv\Scripts\activate
    ```

3.  **Install Dependencies:**
    *Use the `requirements.txt` file we generated to install all necessary packages:*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Monitor:**
    ```bash
    python real_time_monitor.py
    ```
    *(Press **'q'** to quit the application.)*

    ### 🔑 External Assets (Required Models & Reference Environment)

Due to file size constraints and non-portability, these essential assets are hosted externally.

#### 1. Required Model Assets (CRITICAL for running the app)

The **Trained Model** (`security_action_model.pkl`) and the **Label Encoder** (`action_encoder.pkl`) are required to run the final application.

* **Download Link:** [DOWNLOAD PKL ASSETS HERE](https://drive.google.com/drive/folders/1LKJZvXZLxYPf1dPAxKiQRmJpGkoF2EUx?usp=drive_link)
* **Action:** Download all files from this folder and place them directly into your project's root directory.

#### 2. Reference Environment (Advanced Use Only)

The **`venv_cv`** folder, containing the complete local environment backup, is provided for reference or advanced troubleshooting.

* **Download Link:** [VENV_CV REFERENCE FOLDER](https://drive.google.com/drive/folders/17sYG7F3cFY2YYf6Iu61YvGkxTtKFqe3_?usp=drive_link)
* > **Note for Users:** It is highly recommended to use the **`requirements.txt`** file to build a new, clean environment on your own system (the professional method), as this folder may not be portable.
