import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ====================== KNOWLEDGE BASE ======================
knowledge_base = [
    "Addis Ababa is the capital and largest city of Ethiopia, known as the diplomatic capital of Africa.",
    "Ethiopia is one of the oldest countries in the world and the only African nation never colonized.",
    "Injera is the national dish, a spongy flatbread made from teff flour, served with various stews.",
    "Lalibela is famous for its 11 medieval rock-hewn churches, a UNESCO World Heritage Site.",
    "Coffee was discovered in Ethiopia. The country has a rich coffee culture.",
    "AI and tech startups are rapidly growing in Addis Ababa, especially in the Bole area.",
    "Ethiopia follows its own calendar, which is 7-8 years behind the Gregorian calendar.",
]

# Embeddings + Vector DB
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(knowledge_base)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load Model with Quantization
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
    device_map="auto"
)

def chatbot(message, history):
    # Retrieve relevant knowledge
    query_emb = embedder.encode([message])
    _, indices = index.search(query_emb, 3)
    context = "\n".join([knowledge_base[i] for i in indices[0]])
    
    # Build prompt
    prompt = f"""You are a helpful, knowledgeable, and friendly Ethiopian AI assistant.

Context from knowledge base:
{context}

Previous conversation:
{chr(10).join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-3:]] if history else [])}

User: {message}
Assistant:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    return response

# Gradio App
demo = gr.ChatInterface(
    fn=chatbot,
    title="🇪🇹 Ethiopian AI Assistant",
    description="RAG + Memory Powered • Ask anything about Ethiopia, culture, history, food, or technology!",
    theme=gr.themes.Soft(),
    examples=[
        "What is special about Ethiopia?",
        "Tell me about Ethiopian food",
        "What should a tourist know before visiting Addis Ababa?",
        "Explain the history of coffee in Ethiopia."
    ]
)

demo.launch()