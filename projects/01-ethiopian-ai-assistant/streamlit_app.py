import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="Ethiopian AI Assistant", page_icon="🇪🇹")

st.title("🇪🇹 Ethiopian AI Assistant")
st.markdown("**Simple RAG Assistant**")

# Load knowledge
@st.cache_resource
def load_knowledge():
    try:
        with open("knowledge_base.txt", "r", encoding="utf-8") as f:
            text = f.read()
        return [line.strip() for line in text.strip().split("\n\n") if line.strip()]
    except:
        return ["Addis Ababa is the capital of Ethiopia.", "Injera is the national dish."]

knowledge_base = load_knowledge()

# Embeddings
@st.cache_resource
def load_db():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(knowledge_base)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index

embedder, index = load_db()

def chatbot(message):
    query_emb = embedder.encode([message])
    _, indices = index.search(query_emb, 2)
    context = "\n".join([knowledge_base[i] for i in indices[0]])
    return f"Based on knowledge: {context}\n\nThis is a simplified response for testing."

# UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Ethiopia..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})