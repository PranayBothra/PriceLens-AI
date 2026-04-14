import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from pathlib import Path
import joblib

# project root 
BASE_DIR = Path(__file__).resolve().parents[1]
@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / 'artifacts' / 'model.joblib')

@st.cache_resource
def load_meta():
    return joblib.load(BASE_DIR / 'artifacts' / 'meta.joblib')
def clean_feature_name(name):
    name = name.replace('col_tnf__', '')
    name = name.replace('remainder__', '')
    name = name.replace('_', ' ')
    return name.title()

def group_features(df):
    grouped = {}

    for _, row in df.iterrows():
        name = row['Feature']
        impact = row['Impact_log']

        if 'Company' in name:
            key = 'Company'
        elif 'Typename' in name:
            key = 'Type'
        elif 'Cpu' in name:
            key = 'CPU'
        elif 'Gpu' in name:
            key = 'GPU'
        elif 'Opsys' in name:
            key = 'OS'
        elif 'Ram' in name:
            key = 'RAM'
        elif 'Ssd' in name:
            key = 'SSD'
        elif 'Hdd' in name:
            key = 'HDD'
        elif 'Weight' in name:
            key = 'Weight'
        elif 'Ips' in name:
            key = 'IPS Display'
        elif 'Ppi' in name:
            key = 'PPI'
        elif 'Touchscreen' in name:
            key = 'Touchscreen'
        else:
            key = name

        grouped[key] = grouped.get(key, 0) + impact

    grouped_df = pd.DataFrame({
        'Feature': grouped.keys(),
        'Impact_log': grouped.values()
    })

    # Convert to % (correct interpretation)
    grouped_df['Impact_percent'] = (np.exp(grouped_df['Impact_log']) - 1) * 100
 
    return grouped_df.sort_values('Impact_log', ascending=False)

def explain_prediction(pipe, sample):
    model = pipe.named_steps['step2']
    preprocessor = pipe.named_steps['step1']

    sample_transformed = preprocessor.transform(sample)

    log_pred = model.predict(sample_transformed)
    price = np.exp(log_pred)[0]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample_transformed)[0]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]

    base_price = np.exp(base_val)

    feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(f) for f in feature_names]

    df = pd.DataFrame({
        'Feature': feature_names,
        'Impact_log': shap_vals
    })

    df = group_features(df)

    return price, base_price, df