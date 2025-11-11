# Brampton AI Pothole Detector (PoC)

![Status: Prototype](httpsa://img.shields.io/badge/Status-Prototype-blue.svg)

This repository contains a fully functional, end-to-end computer vision prototype built as part of a Proof of Concept (PoC) proposal for the **City of Brampton's "Automated Pothole Detection"** focus area.

The application uses a custom-trained **YOLOv8 (You Only Look Once)** object detection model to identify and track potholes in real-time from video feeds.

---

## 🚀 Live Demo

A live version of this prototype, deployed via Streamlit Community Cloud, can be accessed here:

**[https://bramptonpotholes.streamlit.app](https://bramptonpotholes.streamlit.app)**

*(Note: The live demo is active. You can upload your own road images or video clips to test the model.)*

---

## ✨ Key Features

### 1. Custom-Trained YOLOv8 Model
The core of this project is a `YOLOv8n` model that was fine-tuned on a public, annotated pothole dataset from Kaggle. This demonstrates the "Transfer Learning" workflow, taking a powerful, pre-trained model and specializing it for a specific task.

### 2. Data Conversion Pipeline (VOC to YOLO)
A key part of this project was data wrangling. The source dataset was in **Pascal VOC (XML) format**, which is incompatible with YOLO. The Colab notebook contains a complete pipeline that:
* Parses the `.xml` annotation files.
* Converts the bounding boxes to the required normalized YOLO `.txt` format.
* Reads the `splits.json` file to correctly organize all images and new labels into `train/` and `test/` (used as `val/`) directories.
* Automatically generates the final `data.yaml` file required for training.

### 3. "Live Asset Management Log"
This PoC proves more than just the technology; it proves the *business case*. The Streamlit app includes a "Live Asset Management Log" that mimics the data needed for a real-world system. This feature logs a timestamp for every pothole detected, fulfilling the city's need to move from a reactive "paper-and-logbook" system to a proactive, data-driven one.

---

## 🛠 Tech Stack

* **App Framework:** Streamlit
* **Computer Vision Model:** YOLOv8 (Ultralytics)
* **Video/Image Processing:** OpenCV, Pillow
* **Data Sourcing:** KaggleHub
* **Build/Deployment:** Google Colab, Git, Streamlit Community Cloud

---

## 🏗 Architecture (Computer Vision Pipeline)

1.  **Ingestion:** The Kaggle dataset (`chitholian/annotated-potholes-dataset`) is downloaded.
2.  **Conversion:** The Colab notebook runs the Python-based conversion script to transform all XML annotations into YOLO-formatted `.txt` labels.
3.  **Training (Fine-Tuning):** The `ultralytics` library is used to fine-tune a pre-trained `yolov8n.pt` model on the newly formatted dataset for 25 epochs.
4.  **Output:** The best-performing model (`best.pt`) is saved, renamed to `pothole_model.pt`, and pushed to the GitHub repo.
5.  **Inference (Streamlit App):**
    * The `app.py` loads `pothole_model.pt`.
    * An uploaded video is processed frame-by-frame using `opencv`.
    * Each frame is passed to the YOLO model, which returns bounding box coordinates.
    * `opencv` is used to draw these boxes onto the frame before it's displayed to the user.

---

## 🏭 Project Reproduction (The "Colab Factory")

The entire Streamlit application (`app.py`, `requirements.txt`, `packages.txt`) and the final trained model (`pothole_model.pt`) are generated and deployed from a single, reproducible Google Colab notebook: `BramptonPothole_Factory.ipynb`.

### How to Run This Project:

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/burnsgregm/BramptonPotholes.git](https://github.com/burnsgregm/BramptonPotholes.git)
    cd BramptonPotholes
    ```
2.  **Open the Notebook:**
    Upload and open `BramptonPothole_Factory.ipynb` in Google Colab.
3.  **Add Colab Secrets:**
    * In Colab, click the "Secrets" (🔑) tab and add:
        * `KAGGLE_USERNAME`: Your Kaggle username.
        * `KAGGLE_KEY`: Your Kaggle API key.
        * `GITHUB_TOKEN`: A GitHub Personal Access Token with `repo` permissions.
4.  **Run the Notebook:**
    * Fill in your `GITHUB_EMAIL` in cell `1.5`.
    * Run all cells from top to bottom.
    * The notebook will download the data, convert it, train the model, write the app files, and push the final, deployable app to this GitHub repository.
5.  **Deploy on Streamlit Cloud:**
    * Connect this GitHub repo to your Streamlit Community Cloud account (if you haven't already).
    * The `packages.txt` file will automatically install the necessary Linux libraries (like `libgl1-mesa-glx`) that OpenCV needs to run.
    * The app will automatically build and go live.

---

## 📄 License

This project is open-sourced under the MIT License.
