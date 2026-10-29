# Bank Marketing Classification Project

## a. Problem statement
Predict whether a bank client will subscribe to a term deposit (`deposit`) and compare five classification algorithms through a Streamlit interface.

## b. Dataset description
The Kaggle Bank Marketing Dataset used here contains 11,162 records, 16 input features, and one binary target. It includes demographic, financial, and campaign-related variables. Source: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data

## c. GitHub Repository Link
 `https://github.com/MGC-design/ML_Classification_Models_And_Evaluation_Metrics`

## d. Models used
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine

Run `python train_models.py` to regenerate saved models, `data/test_data.csv`, and `model_comparison.csv`.

***Data Path :*** https://github.com/MGC-design/ML_Classification_Models_And_Evaluation_Metrics/tree/main/data

## Evaluation metrics
Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC). The generated `model_comparison.csv` contains the final measured values.
#### Path : https://github.com/MGC-design/ML_Classification_Models_And_Evaluation_Metrics/blob/main/model_comparison.csv

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.8289 | 0.9071 | 0.8207 | 0.8176 | 0.8191 | 0.6569 |
| Decision Tree | 0.8249 | 0.8926 | 0.7848 | 0.8686 | 0.8246 | 0.6540 |
| K-Nearest Neighbors | 0.8258 | 0.8974 | 0.8342 | 0.7892 | 0.8111 | 0.6505 |
| Gaussian Naive Bayes | 0.7201 | 0.8042 | 0.7837 | 0.5652 | 0.6568 | 0.4472 |
| Support Vector Machine | 0.8554 | 0.9192 | 0.8182 | 0.8932 | 0.8540 | 0.7139 |

## Streamlit features
- Upload labelled test CSV
- Select one of five saved models
- Display six required metrics
- Display confusion matrix and classification report
- Download predictions
- No retraining on uploaded test data

## Local setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Use `data/test_data.csv` for testing the UI.

## Deployment
Push all files to GitHub, open https://share.streamlit.io/, choose the repository and branch, set the main file to `app.py`, and deploy.

## Live Streamlit Application
**App link :** https://mlclassificationmodelsandevaluationmetrics.streamlit.app

## Observations
#### Review `model_comparison.csv`. Higher AUC and MCC indicate stronger ranking and balanced classification. 

| ML Model | Observation |
|-----------|------------|
| Logistic Regression | Logistic Regression is interpretable. |
| Decision Tree | Decision Tree provides nonlinear rules. |
| KNN | KNN is distance-based and prediction-heavy. |
| Gaussian Naive Bayes | Gaussian Naive Bayes is fast but assumes conditional independence. |
| Support Vector Machine | Best overall performer with highest Accuracy, AUC, F1 and MCC. |
| ***Overall Winner*** | ***so in this case <ins>Support Vector Machine(SVM)</ins> is clear winner.*** | 


## Detailed Joblib Creation Process

Run `python train_models.py` to train all models and automatically create the .joblib files.

Example save:
```python
import joblib
joblib.dump(model,"saved_models/logistic_regression.joblib")
```

Example load:
```python
import joblib
model = joblib.load("saved_models/logistic_regression.joblib")
```

Generated files:
- logistic_regression.joblib
- decision_tree.joblib
- knn.joblib
- naive_bayes.joblib
- svm.joblib

  ## Workflow

1. Load Bank Marketing Dataset
2. Preprocess categorical and numerical features
3. Split dataset into training and testing sets
4. Train five classification algorithms
5. Evaluate models using Accuracy, AUC, Precision, Recall, F1 and MCC
6. Save trained models using Joblib
7. Build Streamlit application
8. Deploy application on Streamlit Community Cloud
