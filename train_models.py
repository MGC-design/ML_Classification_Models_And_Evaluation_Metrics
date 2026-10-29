from pathlib import Path
import json, pandas as pd
from sklearn.model_selection import train_test_split
from models import LogisticRegressionModel, DecisionTreeModel, KNNModel, NaiveBayesModel, SVMModel
from utils.evaluation import evaluate_classifier

ROOT=Path(__file__).resolve().parent; TARGET="deposit"
MODEL_CLASSES=[LogisticRegressionModel, DecisionTreeModel, KNNModel, NaiveBayesModel, SVMModel]
MODEL_FILES={"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib",
             "K-Nearest Neighbors":"knn.joblib","Gaussian Naive Bayes":"naive_bayes.joblib",
             "Support Vector Machine":"svm.joblib"}

def main():
    df=pd.read_csv(ROOT/'data'/'bank.csv')
    X=df.drop(columns=TARGET); y=df[TARGET].map({'no':0,'yes':1})
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
    test=X_test.copy(); test[TARGET]=y_test.map({0:'no',1:'yes'}); test.to_csv(ROOT/'data'/'test_data.csv',index=False)
    rows=[]
    for cls in MODEL_CLASSES:
        model=cls().train(X_train,y_train); pred=model.predict(X_test); score=model.predict_score(X_test)
        metrics=evaluate_classifier(y_test,pred,score); rows.append({'ML Model Name':model.model_name,**metrics})
        model.save(ROOT/'saved_models'/MODEL_FILES[model.model_name]); print(model.model_name, metrics)
    pd.DataFrame(rows).to_csv(ROOT/'model_comparison.csv',index=False)
    (ROOT/'saved_models'/'metadata.json').write_text(json.dumps({'target':TARGET,'features':X.columns.tolist(),'model_files':MODEL_FILES},indent=2))
if __name__=='__main__': main()
