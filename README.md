# Diabetes-Prediction-Model-using-Support-Vector-Machine-SVM-
"Predict diabetes using SVM. Trained on PIMA Diabetes Dataset, this model achieves high accuracy. Easily deployable for individual risk assessment."
**Diabetes Prediction Model using Support Vector Machine (SVM)**
## OverView
This repository contains code for a machine learning model designed to predict the likelihood of diabetes based on various health metrics. The model utilizes Support Vector Machine (SVM) classification, a powerful algorithm suitable for binary classification tasks. The dataset used for training and testing the model is the PIMA Diabetes Dataset obtained from the UCI Machine Learning Repository on Kaggle.

## Features
### Data Collection and Analysis:
The dataset is loaded into a pandas dataframe, where statistical summaries and distributions of the features are explored.
### Data Preprocessing:
Features are standardized using the StandardScaler to ensure uniform scaling across the dataset.
Model Training: An SVM classifier with a linear kernel is trained on the standardized data to learn the patterns associated with diabetes.
Model Evaluation: The trained model's performance is evaluated using accuracy scores on both training and testing datasets.
Prediction: Sample input data can be provided to the model to predict the likelihood of diabetes for an individual.
## Instructions
Clone the repository to your local machine.
Ensure you have Python and required libraries installed (NumPy, pandas, scikit-learn).
Run the provided Python script or notebook to execute the model.
Follow the prompts to input data for prediction or modify the code as needed for your use case.
Dataset
The PIMA Diabetes Dataset consists of various health metrics such as glucose level, blood pressure, and body mass index (BMI), along with corresponding labels indicating diabetes status (0: Non-Diabetic, 1: Diabetic).

## Acknowledgments
This project is based on the work of the UCI Machine Learning Repository and Kaggle contributors who provided the PIMA Diabetes Dataset.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
