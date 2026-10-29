from pathlib import Path
import json, joblib, pandas as pd, streamlit as st, matplotlib.pyplot as plt, seaborn as sns
from utils.evaluation import full_evaluation
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Bank Marketing Classification", page_icon="📊", layout="wide")
st.title("Bank Marketing Classification Models")
st.caption("Upload labelled test data, select a saved model, and compare classification performance. Models are loaded without retraining.")
meta=json.loads((ROOT/'saved_models'/'metadata.json').read_text()); target=meta['target']
upload=st.sidebar.file_uploader("Upload test CSV",type=['csv'])
model_name=st.sidebar.selectbox("Select model",list(meta['model_files']))
st.sidebar.info("The CSV must contain all 16 feature columns and the deposit target column.")
if upload is None:

    st.info("Upload data/test_data.csv to begin.")

    if (ROOT / "model_comparison.csv").exists():

        comparison_df = pd.read_csv(
            ROOT / "model_comparison.csv"
        )

        selected = comparison_df[
            comparison_df["ML Model Name"] == model_name
        ].iloc[0]

        st.subheader(f"{model_name} Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{selected['Accuracy']:.4f}"
        )

        col2.metric(
            "AUC",
            f"{selected['AUC']:.4f}"
        )

        col3.metric(
            "Precision",
            f"{selected['Precision']:.4f}"
        )

        col1.metric(
            "Recall",
            f"{selected['Recall']:.4f}"
        )

        col2.metric(
            "F1",
            f"{selected['F1']:.4f}"
        )

        col3.metric(
            "MCC",
            f"{selected['MCC']:.4f}"
        )

else:
    try:
        df=pd.read_csv(upload); required=set(meta['features']+[target]); missing=sorted(required-set(df.columns))
        if missing: st.error("Missing required columns: "+", ".join(missing)); st.stop()
        X=df[meta['features']]; y=df[target].astype(str).str.lower().map({'no':0,'yes':1})
        if y.isna().any(): st.error("The deposit column must contain only yes/no values."); st.stop()
        pipe=joblib.load(ROOT/'saved_models'/meta['model_files'][model_name]); pred=pipe.predict(X)
        score=pipe.predict_proba(X)[:,1] if hasattr(pipe,'predict_proba') else pipe.decision_function(X)
        metrics,cm,report=full_evaluation(y,pred,score)
        st.subheader(model_name); cols=st.columns(6)
        for col,(name,value) in zip(cols,metrics.items()): col.metric(name,f"{value:.4f}")
        left,right=st.columns(2)
        with left:
            st.markdown("#### Confusion matrix"); fig,ax=plt.subplots(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,xticklabels=['No','Yes'],yticklabels=['No','Yes'],ax=ax); ax.set(xlabel='Predicted',ylabel='Actual'); st.pyplot(fig); plt.close(fig)
        with right:
            st.markdown("#### Classification report"); st.dataframe(pd.DataFrame(report).T.style.format(precision=4),use_container_width=True)
        out=df.copy(); out['predicted_deposit']=pd.Series(pred).map({0:'no',1:'yes'}); out['deposit_probability']=score
        st.download_button("Download predictions",out.to_csv(index=False).encode(),"predictions.csv","text/csv")
    except Exception as exc: st.exception(exc)
