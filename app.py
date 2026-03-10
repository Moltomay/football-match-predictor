import streamlit as st
import json
import os
from backend.runner import run_prediction

st.set_page_config(page_title="Football Match Predictor", layout="wide")

st.title("⚽ Football Match Result Predictor")
st.markdown("""
Enter two football teams to get a data-driven prediction based on recent form, 
head-to-head history, and league standings.
""")

col1, col2 = st.columns(2)

with col1:
    team_a = st.text_input("Team A (Home/Favored)", placeholder="e.g. Arsenal")

with col2:
    team_b = st.text_input("Team B (Away/Challenger)", placeholder="e.g. Manchester City")

if st.button("Predict Outcome", type="primary"):
    if not team_a or not team_b:
        st.error("Please enter both team names.")
    else:
        with st.spinner(f"Analyzing {team_a} vs {team_b}..."):
            try:
                prediction = run_prediction(team_a, team_b)
                
                # Display Results
                st.success("Analysis Complete!")
                
                # Prediction Summary
                st.header(f"Prediction: {prediction.get('winner', 'N/A')}")
                
                p_val = prediction.get('probability', 0)
                st.progress(p_val / 100.0)
                st.write(f"**Confidence:** {p_val}%")
                
                st.subheader("Reasoning")
                st.write(prediction.get("reasoning", "No reasoning provided."))
                
                st.subheader("Top Signals")
                for signal in prediction.get("top_signals", []):
                    st.write(f"- {signal}")

                # Detailed Transparency
                with st.expander("View Full Analysis Data"):
                    # Load latest log
                    log_path = prediction.get("log_path")
                    if log_path and os.path.exists(log_path):
                        with open(log_path, "r", encoding="utf-8") as f:
                            log_data = json.load(f)
                        
                        st.json(log_data)
                    else:
                        st.warning("Log file not found.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.info("Check if your TAVILY_API_KEY and MODEL_PATH are set correctly in .env")

st.sidebar.header("About")
st.sidebar.info("""
This tool uses a local LLM and real-time web search (Tavily) to gather and analyze 
match data. It's intended as a prototype for technical demonstration.
""")
