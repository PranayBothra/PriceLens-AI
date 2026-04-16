import requests
import os
from dotenv import load_dotenv
from streamlit import secrets

try:
    API_KEY = secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")

def generate_text_explanation(df):
    top_pos = df[df['Impact_log'] > 0].sort_values('Impact_log', ascending=False).head(3)
    top_neg = df[df['Impact_log'] < 0].sort_values('Impact_log').head(3)

    explanation = "Key factors:\n"

    for _, row in top_pos.iterrows():
        explanation += f"• {row['Feature']} increased price by {row['Impact_percent']:.1f}%\n"

    for _, row in top_neg.iterrows():
        explanation += f"• {row['Feature']} decreased price by {abs(row['Impact_percent']):.1f}%\n"

    return explanation


def generate_ai_insights(df, price, base_price):
    # CASCADING FALLBACK ARRAYS
    gemini_models = [
        "gemini-3-flash",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite"
    ]
    
    gemma_models = [
        "gemma-4-31b",
        "gemma-4-26b",
        "gemma-3-27b",
        "gemma-3-12b"
    ]

    # Prepare structured input
    top_pos = df[df['Impact_log'] > 0].sort_values('Impact_log', ascending=False).head(3)
    top_neg = df[df['Impact_log'] < 0].sort_values('Impact_log').head(3)

    pos_string = top_pos[['Feature','Impact_percent']].to_string(index=False) if not top_pos.empty else "None"
    neg_string = top_neg[['Feature','Impact_percent']].to_string(index=False) if not top_neg.empty else "None"

    # --- 1. GEMINI PROMPT (Original, nuanced prompt) ---
    gemini_prompt = f"""
    You are an expert data analyst explaining a machine learning prediction.

    Context:
    - Average laptop price: ₹{int(base_price)}
    - Predicted price: ₹{int(price)}

    Top factors increasing price:
    {pos_string}

    Top factors decreasing price:
    {neg_string}

    Your task:
    1. Explain WHY the price is higher/lower than average
    2. Highlight the MOST important factors
    3. Keep explanation simple and natural (like explaining to a user)
    4. Do NOT mention SHAP, log, or technical terms

    Give a clean paragraph + bullet points.
    """

    # --- 2. GEMMA PROMPT (Strict, Rule-Based Template) ---
    # Gemma needs very explicit formatting rules to avoid hallucinating
    gemma_prompt = f"""
    Write exactly 3-4 bullet points explaining a laptop price. Do not write paragraphs or intros.

    Context:
    Average laptop price: ₹{int(base_price)}
    Predicted price: ₹{int(price)}
    Increasing factors: {pos_string}
    Decreasing factors: {neg_string}

    Rules:
    - Bullet 1: State the predicted price and whether it is higher or lower than the average.
    - Bullet 2: Explain the main features that increased the price.
    - Bullet 3: Explain the main features that kept the price down.
    - Do NOT mention SHAP, machine learning, or logs. Use simple language.
    """

    # --- 3. API CALL FUNCTION ---
    def call_api(model_name, prompt_text):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        response = requests.post(url, json=payload)
        return response.json()

    # --- 4. EXECUTION LOOP (Gemini first, then Gemma) ---
    
    # Phase 1: Try Gemini
    for model in gemini_models:
        try:
            data = call_api(model, gemini_prompt)
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif "error" in data:
                err_msg = data['error']['message'].lower()
                if "quota" in err_msg or "limit" in err_msg or data['error'].get('code') == 429:
                    continue # Quota exhausted, try next Gemini
        except Exception:
            continue

    # Phase 2: Try Gemma if all Gemini models fail
    for model in gemma_models:
        try:
            data = call_api(model, gemma_prompt)
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif "error" in data:
                err_msg = data['error']['message'].lower()
                if "quota" in err_msg or "limit" in err_msg or data['error'].get('code') == 429:
                    continue # Quota exhausted, try next Gemma
        except Exception:
            continue

    # Ultimate fallback if everything is down
    return "⚠️ High traffic detected. AI insights are currently unavailable. Please rely on the numerical metrics provided above."