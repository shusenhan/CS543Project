YOLOv5 Real-Time Lane Detection Project

This project aims to perform real-time lane detection using YOLOv5. We utilize the CULane dataset to train the model to detect lane lines accurately and efficiently. The project is built on PyTorch and uses dice_loss as a loss function to improve segmentation performance.

Table of Contents

	•	Project Overview
	•	Features
	•	Installation
	•	Dataset Preparation
	•	Training
	•	Testing
	•	File Structure
	•	Known Issues
	•	Contributing

Project Overview

This project leverages the YOLOv5 model combined with image segmentation and lane detection algorithms. It is trained using labeled lane data, enabling it to identify and predict lane lines in images, making it suitable for applications in autonomous and assisted driving systems.

Features

	•	Real-time lane detection
	•	Trained and validated on the CULane dataset
	•	Integrated dice_loss for improved segmentation accuracy
	•	Built with PyTorch and torchvision for training and testing

Installation

1. Clone the Repository

git clone https://github.com/your-username/YoloV5-LaneDetection.git
cd YoloV5-LaneDetection

2. Install Dependencies

conda create -n yolov5_env python=3.8
conda activate yolov5_env
pip install -r requirements.txt

The requirements.txt file includes torch, torchvision, and other necessary libraries.

3. Configure Dataset Paths

Place the CULane dataset in the CULane/ directory and ensure the dataset file structure matches the project requirements.

Dataset Preparation

	•	Download the CULane dataset and extract it into the CULane/ folder.
	•	Ensure that file paths match those specified in main.py.

Training

Start Training

python main.py

The training process will output loss values, progress, and other relevant metrics in the command line. main.py will automatically load the dataset and configure data augmentation and training parameters.

Testing

python test.py

The test script loads the trained model and runs it on the test set, generating predictions. Results can be found in the outputs/ directory.

File Structure

	•	main.py: Main script for model training.
	•	test.py: Script for model testing.
	•	model.py: YOLOv5 model and custom loss function.
	•	CULane/: Directory containing the CULane dataset.
	•	outputs/: Directory for saving model outputs (e.g., lane detection results).

Known Issues

	•	Size Mismatch Error: RuntimeError: The size of tensor a must match the size of tensor b. Ensure that input image dimensions are consistent and that labels and predictions have matching channels.
	•	PyTorch Version Compatibility: Recommended versions are torch>=1.7 and torchvision>=0.8.

Contributing

We welcome issues and pull requests to help improve the project. Please test all functionality before submitting code.

If you have any questions, feel free to reach out through the project page. Thank you for your interest and contributions!

Feel free to adjust this README template according to your project’s specific requirements.
