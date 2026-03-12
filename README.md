# LSTM-Based Baseball Pitch Prediction Model

# Project Overview
This project was created as a capstone assignment for ISC353 Topics in Information Science course at International Christian University (ICU). It aims to create a machine learning model that predicts the different pitching types(Curveball, Fastball, Slider) based on the pre-pitch position of the pitcher extracted from multiple throwing clips. Using the motion landmarks, Long Short-Term Memory (LSTM) was then trained to track the temporal dependencies of the pitcher's movement sequence and classify the pitching type.

# Hypothesis
A pitcher's pre-throw motion consists of temporal patterns that can be used to predict the pitch type prior to throwing. 

# Dataset
The dataset was manually created and consists of pitching videos used to extract pose landmarks which represents the body's movements overtime. Each sample contains data
on the corresponding pitching types:
- Curveballs
- Fastballs
- Sliders

Dataset access: https://drive.google.com/drive/folders/1hXhuYaFO7BRnTka2cpRQDer6cmQ8bMcg?usp=sharing

# Results Summary
- The LSTM model successfully identified intricate details in the pitching motion.
- The model achieved consistent prediction accuracy across validation samples, with clear accuracy in Fastballs and Curveballs. 
Example Plots: Training vs Validation Loss, Prediction Accuracy by Pitch Type

# Team Contribution
**Davian Buana**
- Implemented the neural network architecture using PyTorch.
- Trained and evaluated the model.
- Summarized the evaluation outcomes via visualization using confusion matrix and bar graphs.

**Mateo Henriquez**
- Designed the overall machine learning pipeline.
- Experimented with other models during the model selection stage.

**Jung Hyun Park**
- Collecting the pitching data from multiple videos.
- Developed the pipeline for converting pitching videos into structured datasets.
- Extracted 33 body landmarks per frame and exported them as CSV files ready for training. 
