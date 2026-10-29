# Bank Marketing Classification Project

## a. Problem statement
Predict whether a bank client will subscribe to a term deposit (`deposit`) and compare five classification algorithms through a Streamlit interface.

## b. Dataset description
The Kaggle Bank Marketing Dataset used here contains 11,162 records, 16 input features, and one binary target. It includes demographic, financial, and campaign-related variables. Source: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data

## c. GitHub Repository Link
Replace after publishing: `https://github.com/MGC-design/ML_Classification_Models_And_Evaluation_Metrics.git`

## d. Models used
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine

Run `python train_models.py` to regenerate saved models, `data/test_data.csv`, and `model_comparison.csv`.

## Evaluation metrics
Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC). The generated `model_comparison.csv` contains the final measured values.

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

## Observations
Review `model_comparison.csv`. Higher AUC and MCC indicate stronger ranking and balanced classification. Logistic Regression is interpretable; Decision Tree provides nonlinear rules; KNN is distance-based and prediction-heavy; Gaussian Naive Bayes is fast but assumes conditional independence; SVM captures nonlinear boundaries but takes longer to train. Declare the model with the strongest test AUC/MCC combination as the overall winner.


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
