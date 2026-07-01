import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ====================== LOAD KNOWLEDGE BASE ======================
def load_knowledge_base(file_path="knowledge_base.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        knowledge = [line.strip() for line in text.strip().split("\n\n") if line.strip()]
        print(f"✅ Loaded {len(knowledge)} knowledge entries")
        return knowledge
    except:
        print("⚠️ knowledge_base.txt not found. Using default.")
        return ["General job search advice.", "Python and ML skills are in high demand."]

knowledge_base = load_knowledge_base()

# Embeddings + Vector DB
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(knowledge_base)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load Model
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

def retrieve_knowledge(query):
    query_emb = embedder.encode([query])
    _, indices = index.search(query_emb, 2)
    return "\n".join([knowledge_base[i] for i in indices[0]])

def analyze_resume(resume_text, job_description):
    context = retrieve_knowledge(job_description)
    prompt = f"""You are a professional career coach.

Context from job market knowledge:
{context}

Resume:
{resume_text}

Job Description:
{job_description}

Provide detailed analysis and suggestions."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs.input_ids, max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio App (same as before)
with gr.Blocks(title="AI Job Application Helper") as demo:
    gr.Markdown("# 🤖 AI Job Application Helper")
    gr.Markdown("Resume Analyzer + Cover Letter Generator")

    with gr.Tab("Resume Analyzer"):
        resume_input = gr.Textbox(label="Paste your Resume", lines=12)
        job_input = gr.Textbox(label="Paste Job Description", lines=6)
        analyze_btn = gr.Button("Analyze Resume")
        output = gr.Textbox(label="Analysis & Suggestions", lines=15)
        analyze_btn.click(analyze_resume, inputs=[resume_input, job_input], outputs=output)

demo.launch(share=True)