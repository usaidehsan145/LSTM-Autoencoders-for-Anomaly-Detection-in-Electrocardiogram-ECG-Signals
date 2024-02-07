# LSTM Autoencoders for Anomaly Detection in Electrocardiogram (ECG) Signals

## Overview
This Python notebook demonstrates the implementation of LSTM autoencoders for anomaly detection using PyTorch. The model is trained on the ECG5000 dataset, which contains electrocardiogram (ECG) recordings with different types of heartbeats. The goal is to train an autoencoder model to reconstruct normal heartbeats accurately and identify anomalies (abnormal heartbeats) based on reconstruction errors.

## Notebook Structure
1. **Dataset Loading**: The ECG5000 dataset is loaded from `.arff` files and preprocessed for training.
2. **Model Architecture**: The autoencoder model architecture consists of an encoder and a decoder implemented using LSTM layers in PyTorch.
3. **Training**: The model is trained using training and validation datasets, and the training progress is monitored using loss values.
4. **Evaluation**: The trained model is evaluated on test datasets containing normal and anomaly heartbeats.
5. **Results Analysis**: The reconstruction losses are visualized to identify anomalies, and the performance of the model in detecting anomalies is evaluated.

## Key Components
- **Encoder**: LSTM layers are used to encode the input sequences into a lower-dimensional representation.
- **Decoder**: LSTM layers decode the encoded representation to reconstruct the input sequences.
- **Training**: The model is trained using L1 loss (mean absolute error) between input and reconstructed sequences.
- **Evaluation**: Reconstruction losses are calculated for both normal and anomaly sequences, and a threshold is applied to classify anomalies.

## Results
- The model demonstrates good performance in reconstructing normal heartbeats with low reconstruction errors.
- Anomalies (abnormal heartbeats) are detected based on higher reconstruction errors compared to normal heartbeats.

## Future Work
- Experiment with different hyperparameters and architectures to improve model performance.
- Explore other anomaly detection techniques and compare their performance with LSTM autoencoders.
