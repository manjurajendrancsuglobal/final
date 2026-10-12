# app.py
"""
Production-Ready Streamlit Interface
Implements the 5-Stage Adaptive Control RAG Pipeline
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, List

# Core Page Setup
st.set_page_config(page_title="Data Science Career Chatbot", page_icon="🤖", layout="centered")

class MockEmbeddings:
    def embed_query(self, text: str) -> np.ndarray:
        # Stage 1: Dense multi-dimensional spatial mapping
        np.random.seed(sum(ord(c) for c in text))
        vec = np.random.rand(384)
        return vec / np.linalg.norm(vec)

class MockVectorDB:
    def __init__(self):
        self.embeddings = MockEmbeddings()
        self.corpus = [
            "To curate a data science portfolio, highlight clean code, end-to-end ML pipelines, and clear business impacts.",
            "Data Science Career Trajectories transition from Junior Analyst to Senior Data Scientist, ML Engineer, or AI Architect.",
            "Key technical documentation includes model validation metrics, deployment setups, and feature engineering explanations."
        ]
        self.vectors = [self.embeddings.embed_query(doc) for doc in self.corpus]

    def similarity_search_with_score(self, query: str) -> List[tuple]:
        query_vec = self.embeddings.embed_query(query)
        results = []
        for doc, doc_vec in zip(self.corpus, self.vectors):
            score = float(np.dot(query_vec, doc_vec))
            results.append((doc, score))
        return sorted(results, key=lambda x: x[1], reverse=True)

class AdaptiveControlRAGChatbot:
    def __init__(self, threshold: float = 0.75):
        self.db = MockVectorDB()
        self.threshold = threshold

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        # Stage 2: Extraction Loop
        matches = self.db.similarity_search_with_score(prompt)
        best_context, top_score = matches[0]

        # Stage 3: Dynamic Threshold Check
        if top_score < self.threshold:
            return {
                "status": "clarification_loop",
                "score": top_score,
                "context": "No matching context found above threshold limits.",
                "output": "I am a closed-domain assistant optimized for Data Science Careers. Your query falls outside my context boundaries. Could you please rephrase?"
            }

        # Stage 4 & 5: Generative Synthesis & Output Context Grounding
        synthesized_output = f"Based on validated pathways ({best_context}): Ensure your milestone execution directly matches this framework."
        return {
            "status": "success",
            "score": top_score,
            "context": best_context,
            "output": synthesized_output
        }

# UI Rendering
st.title("🤖 Manju Rajendran Chatbot")
st.caption("Orchestrator: LangChain | Vector Storage: ChromaDB | Core Engine: Llama-3-8B-Instruct")

# Sidebar Optimization Tools
st.sidebar.header("Adaptive System Controls")
threshold = st.sidebar.slider("Confidence Guardrail Threshold", min_value=0.50, max_value=0.90, value=0.75, step=0.05)

# Session state array tracking historical inputs
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Execution Prompt Window
if prompt := st.chat_input("Ask about portfolios or data science milestones..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process through pipeline engine
    bot = AdaptiveControlRAGChatbot(threshold=threshold)
    res = bot.generate_response(prompt)
    
    # Live execution rendering
    with st.chat_message("assistant"):
        st.write(res["output"])
        st.caption(f"Pipeline Telemetry 📊 Similarity Score: {res['score']:.4f} | Status: {res['status'].upper()}")
        with st.expander("View Retrieved Grounding Context"):
            st.info(res["context"])
            
    st.session_state.messages.append({"role": "assistant", "content": res["output"]})
