# Title: LSTM-Based Baseball Pitch Prediction Model

# Project Overview
This project was a capstone assignment for ISC353 Topics in Information Science course at International Christian University (ICU) developed by a team of student developers. It aims to create a machine learning model that predicts the pithing types(Curveball, Fastball, Slider) based on the pre-pitch movement of the pitcher. Through analyzing the motion landmarks extracted from the videos, the LSTM (Long Short-Term Memory) neural network is trained to recognize the temporal patterns of the pitching types.

# Hypothesis
We can predict the type of pitch a pitcher will throw based on their pre-pitch motion.

# Requirements
Install these dependencies prior to running: pip install torch torchvision mediapipe numpy pandas matplotlib
Link to the dataset: https://drive.google.com/drive/folders/1hXhuYaFO7BRnTka2cpRQDer6cmQ8bMcg?usp=sharing

# Run
python pitch_prediction.py

# Results Summary
- The LSTM model successfully identified intricate details in the pitching motion.
- The model achieved consistent prediction accuracy across validation samples, with clear accuracy in Fastballs and Curveballs. 
Example Plots: Training vs Validation Loss, Prediction Accuracy by Pitch Type

# Team Contribution
- Davian Buana: Implemented the LSTM model using the PyTorch library, evaluated the accuracy through visualization via confusion matrix and bar graphs.
- Mateo Henriquez: Designed the machine learning pipeline, assisted in the selection of the LSTM model after experimenting with various models. 
- Jung Hyun Park: Collected and organized the datasets, took charge of the preprocessing and data cleaning. 
