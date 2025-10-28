# 🌟 Proactive Real-Time Behavioral Security Monitoring System

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
    git clone [YOUR-REPO-URL]
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
