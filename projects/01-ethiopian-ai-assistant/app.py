import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ====================== LOAD KNOWLEDGE FROM FILE ======================
def load_knowledge_base(file_path="knowledge_base.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        knowledge_base = [line.strip() for line in text.strip().split("\n\n") if line.strip()]
        print(f"✅ Loaded {len(knowledge_base)} knowledge entries from {file_path}")
        return knowledge_base
    except FileNotFoundError:
        print(f"⚠️ {file_path} not found. Using default knowledge.")
        return [
            "Addis Ababa is the capital and largest city of Ethiopia.",
            "Ethiopia is known as the cradle of humanity.",
            "Injera is the national dish of Ethiopia.",
            "Lalibela is famous for its rock-hewn churches.",
            "Coffee was discovered in Ethiopia.",
        ]

knowledge_base = load_knowledge_base()

# ====================== EMBEDDINGS + VECTOR DB ======================
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(knowledge_base)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ====================== LOAD MODEL ======================
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
    
    # Build prompt with history
    history_text = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-4:]] if history else [])
    
    prompt = f"""You are a helpful, knowledgeable, and friendly Ethiopian AI assistant.

Context from knowledge base:
{context}

Recent conversation:
{history_text}

User: {message}
Assistant:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=350,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    return response

# ====================== GRADIO APP ======================
demo = gr.ChatInterface(
    fn=chatbot,
    title="🇪🇹 Ethiopian AI Assistant",
    description="RAG + Memory Powered • Ask anything about Ethiopia, culture, history, food, or technology!",
    examples=[
        "What is special about Ethiopia?",
        "Tell me about Ethiopian food and culture.",
        "What should a tourist know before visiting Addis Ababa?",
        "Explain the history of coffee in Ethiopia."
    ]
)

demo.launch(share=True)