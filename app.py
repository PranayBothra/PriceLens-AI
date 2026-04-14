import streamlit as st
import pandas as pd

from src.ml_core import load_model, load_meta, explain_prediction
from src.visualization import plot_feature_impact
from src.insights import generate_text_explanation, generate_ai_insights

# PAGE CONFIG
st.set_page_config(
    page_title="PriceLens AI",
    page_icon="💻",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL + META (with spinner)
with st.spinner("Loading model..."):
    try:
        pipe = load_model()
        meta = load_meta()
    except FileNotFoundError:
        st.error("🚨 Model files not found. Check paths in ml_core.py")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Failed to load model: {e}")
        st.stop()

# SESSION STATE
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.price = None
    st.session_state.base_price = None

# TITLE
st.markdown("""
<h1 style='text-align:center;'>💻 PriceLens AI</h1>
<p style='text-align:center; color:gray;'>
From Raw Data → Explainable Insights → AI Explanation
</p>
""", unsafe_allow_html=True)

st.divider()

# INPUT SECTION
st.subheader("🛠 Configure Your Laptop")

with st.form("laptop_config_form"):
    cols = st.columns(3)
    input_data = {}

    i = 0
    for col, info in meta.items():
        with cols[i % 3]:
            
            # 🔥 Unit formatting logic (Sirf UI ke liye)
            display_label = col
            col_lower = col.lower()
            
            if 'ram' in col_lower or 'ssd' in col_lower or 'hdd' in col_lower:
                display_label = f"{col} (GB)"
            elif 'weight' in col_lower:
                display_label = f"{col} (kg)"
            elif 'clock' in col_lower or 'cpu' in col_lower and 'freq' in col_lower:
                display_label = f"{col} (GHz)"

            # 🔥 Input Widgets
            if info['type'] == 'categorical':
                # Dropdown
                input_data[col] = st.selectbox(display_label, info['values'])

            elif info['type'] == 'binary':
                # Toggle switch
                val = st.toggle(display_label, value=False)
                input_data[col] = 1 if val else 0

            else:
                # 👉 Sirf Number Input (No sliders)
                # float() aur step=0.1 lagaya hai taaki decimals (jaise 1.5 kg) aaram se type ho sakein
                input_data[col] = st.number_input(
                    display_label,
                    min_value=float(info['min']),
                    max_value=float(info['max']),
                    value=float(info['min']),
                    step=0.1 
                )

        i += 1

    st.divider()
    submitted = st.form_submit_button("🚀 Predict & Explain", use_container_width=True)

# MAIN EXECUTION
if submitted:

    # clear old AI output
    if "ai_text" in st.session_state:
        del st.session_state["ai_text"]

    input_df = pd.DataFrame([input_data])

    with st.spinner("Analyzing laptop configuration..."):
        try:
            price, base_price, df = explain_prediction(pipe, input_df)

            st.session_state.df = df
            st.session_state.price = price
            st.session_state.base_price = base_price

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

# RESULTS
if st.session_state.df is not None:

    price = st.session_state.price
    base_price = st.session_state.base_price
    df = st.session_state.df

    #METRICS
    col1, col2, col3 = st.columns(3)

    diff = price - base_price

    col1.metric("💰 Predicted Price", f"₹ {int(price)}")
    col2.metric("📊 Average Price", f"₹ {int(base_price)}")
    col3.metric("📈 Difference", f"₹ {int(diff)}")

    st.divider()

    # GRAPH + TABLE
    colA, colB = st.columns([1.2, 1])

    with colA:
        st.subheader("📊 Feature Impact")
        try:
            with st.spinner("Generating visualization..."):
                fig = plot_feature_impact(df)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Visualization error: {e}")

    with colB:
        st.subheader("📋 Top Factors")
        display_df = df[['Feature', 'Impact_percent']].head(10)
        
        st.dataframe(display_df, use_container_width=True)

    st.divider()

    #BASIC EXPLANATION
    st.subheader("🧠 Key Insights")
    st.markdown(generate_text_explanation(df))

    # AI EXPLANATION
    st.divider()
    st.subheader("🤖 AI Explanation")

    if st.button("✨ Generate AI Explanation"):

        with st.spinner("Generating AI insights..."):
            try:
                st.session_state.ai_text = generate_ai_insights(
                    df,
                    price,
                    base_price
                )
            except Exception as e:
                st.session_state.ai_text = f"⚠️ API Error: {e}"

    if "ai_text" in st.session_state:

        ai_text = st.session_state.ai_text

        if "⚠️" in ai_text:
            st.error(ai_text)
        else:
            st.markdown(f"""
            <div style='background:#1c1f26;padding:15px;border-radius:10px'>
            {ai_text}
            </div>
            """, unsafe_allow_html=True)