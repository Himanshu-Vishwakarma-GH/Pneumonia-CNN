# Pneumonia Detection with CNN & Grad-CAM

This project provides a complete deep learning pipeline for detecting pneumonia from chest X-ray images. It demonstrates how to transition an AI model from experimental notebooks into a deployable web application, featuring robust explainability using Grad-CAM.

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Deploy-orange.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Himanshu-Vishwakarma-HF/Pneumonia-CNN)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Himanshu-Vishwakarma-HF/Pneumon)

## Features

- **Dual Model Architecture**: Choose between training a baseline CNN from scratch or utilizing Transfer Learning with a pre-trained **ResNet18** model.
- **Resumable Training**: The training loop automatically checkpoints the model state after every epoch (`resnet_checkpoint.pth`). If training is interrupted, simply run the cell again to seamlessly resume from exactly where it left off.
- **Explainable AI (XAI)**: Integrated `pytorch-grad-cam` to generate heatmaps highlighting the exact regions of the lung the model focused on when making its prediction.
- **Live Web Demo**: A standalone `app.py` built with Gradio, ready to be deployed directly to Hugging Face Spaces for live inference.

## Project Structure

```text
├── pneumonia_detection.ipynb  # Core training workspace (Data Prep, Models, Checkpointing, Grad-CAM)
├── app.py                     # Standalone Gradio web interface for deployment
├── requirements.txt           # Python dependencies
└── .gitignore                 # Excludes dataset and large model checkpoints
```
*(Note: The X-ray dataset and `.pth` checkpoints are ignored by git due to size constraints).*

## Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Himanshu-Vishwakarma-GH/Pneumonia-CNN.git
cd Pneumonia-CNN
pip install -r requirements.txt
```

### 2. Training the Model
1. Place the dataset inside a folder named `xray_dataset_covid19/` (containing `train` and `test` subfolders).
2. Open `pneumonia_detection.ipynb` in Jupyter or VS Code.
3. Run the cells to initiate the resumable training loop. It will automatically save an epoch checkpoint (`resnet_checkpoint.pth` or `cnn_checkpoint.pth`).

### 3. Running the Live Demo
Once you have generated a checkpoint file from the notebook, you can launch the Gradio web interface:
```bash
python app.py
```
This will open a local web server (typically `http://127.0.0.1:7860`) where you can drag and drop X-rays and instantly view the AI's prediction alongside the Grad-CAM heatmap.

## Technologies Used
- PyTorch & Torchvision
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Gradio (Web Deployment)
- OpenCV & PIL (Image processing)
