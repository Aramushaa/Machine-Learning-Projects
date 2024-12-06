Anomaly Detection Using Autoencoders and GANs

This repository contains my anomaly detection project for university, where I used autoencoders and Generative Adversarial Networks (GANs) to identify anomalies. The project showcases how deep learning models can effectively detect outliers in data, which can be used in various real-world applications such as fraud detection, fault diagnosis, and medical imaging.

Project Overview

Anomaly detection is a crucial problem in machine learning, and this project aims to leverage advanced neural network architectures to detect anomalies in data. Specifically, the project involves:

Autoencoders: Used to reconstruct normal data patterns, allowing anomalies to be detected when reconstruction error exceeds a predefined threshold.

Generative Adversarial Networks (GANs): Employed to generate synthetic data and identify anomalies by distinguishing between normal and anomalous patterns.

The combination of autoencoders and GANs enhances the model's ability to learn the underlying distribution of normal data, making it more effective at detecting unusual patterns.

Technologies Used

Python: The main programming language for building and training models.

TensorFlow / PyTorch: Used for building and training the autoencoder and GAN models.

NumPy & Pandas: For data preprocessing and handling.

Matplotlib : For data visualization and analysis.

Project Structure

data/: Contains datasets used for training and testing.

models/: Includes the implementation of autoencoders and GAN models.

notebooks/: Jupyter notebooks for experimenting, visualizing results, and explaining the steps.

scripts/: Python scripts for training and evaluating models.

results/: Saved models 