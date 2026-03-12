# Title: LSTM-Based Baseball Pitch Prediction Model

# Project Overview
This project was created as a capstone assignment for ISC353 Topics in Information Science course at International Christian University (ICU). It aims to create a machine learning model that predicts the different pitching types(Curveball, Fastball, Slider) based on the pre-pitch position of the pitcher extracted from multiple throwing clips. Using the motion landmarks, Long Short-Term Memory (LSTM) was then trained to track the temporal dependencies of the pitcher's movement sequence and classify the pitching type.

# Hypothesis
A pitcher's pre-throw motion consists of temporal patterns that can be used to predict the pitch type prior to throwing. 

# Requirements
Necessary dependencies: pip install torch torchvision mediapipe numpy pandas matplotlib
Dataset access: https://drive.google.com/drive/folders/1hXhuYaFO7BRnTka2cpRQDer6cmQ8bMcg?usp=sharing

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
