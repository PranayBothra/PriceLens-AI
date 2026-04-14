import requests
import os
from dotenv import load_dotenv

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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

    # Prepare structured input
    top_pos = df[df['Impact_log'] > 0].sort_values('Impact_log', ascending=False).head(3)
    top_neg = df[df['Impact_log'] < 0].sort_values('Impact_log').head(3)

    prompt = f"""
You are an expert data analyst explaining a machine learning prediction.

Context:
- Average laptop price: ₹{int(base_price)}
- Predicted price: ₹{int(price)}

Top factors increasing price:
{top_pos[['Feature','Impact_percent']].to_string(index=False)}

Top factors decreasing price:
{top_neg[['Feature','Impact_percent']].to_string(index=False)}

Your task:
1. Explain WHY the price is higher/lower than average
2. Highlight the MOST important factors
3. Keep explanation simple and natural (like explaining to a user)
4. Do NOT mention SHAP, log, or technical terms

Give a clean paragraph + bullet points.
"""

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload)
        data = response.json()

        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in data:
            error_msg = data['error']['message'].lower()

            if "quota" in error_msg or "limit" in error_msg:
                return "⚠️ AI quota exhausted. Please try again after some time."

            return f"API Error: {data['error']['message']}"
        else:
            return "Unexpected response from AI."

    except Exception:
        return "⚠️ AI service is currently unavailable. Please try again later."