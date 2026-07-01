import gradio as gr
from transformers import pipeline
import torch

# Load Model
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=400,
    temperature=0.7
)

def analyze_resume(resume_text, job_description):
    prompt = f"""You are a professional career coach specializing in tech jobs.

Resume:
{resume_text}

Job Description:
{job_description}

Provide:
1. Strengths
2. Weaknesses
3. Suggestions for improvement
4. Key skills to highlight

Answer:"""

    response = pipe(prompt)[0]['generated_text']
    return response

def generate_cover_letter(resume_text, job_description):
    prompt = f"""Write a professional, tailored cover letter based on the resume and job description.

Resume:
{resume_text}

Job Description:
{job_description}

Cover Letter:"""

    response = pipe(prompt)[0]['generated_text']
    return response

# Gradio Interface
with gr.Blocks(title="AI Job Application Helper") as demo:
    gr.Markdown("# 🤖 AI Job Application Helper")
    gr.Markdown("Resume Analyzer + Cover Letter Generator + Interview Prep")

    with gr.Tab("Resume Analyzer"):
        resume_input = gr.Textbox(label="Paste your Resume", lines=10)
        job_input = gr.Textbox(label="Paste Job Description", lines=5)
        analyze_btn = gr.Button("Analyze Resume")
        output = gr.Textbox(label="Analysis & Suggestions", lines=15)
        analyze_btn.click(analyze_resume, inputs=[resume_input, job_input], outputs=output)

    with gr.Tab("Cover Letter Generator"):
        resume2 = gr.Textbox(label="Paste your Resume", lines=8)
        job2 = gr.Textbox(label="Paste Job Description", lines=5)
        generate_btn = gr.Button("Generate Cover Letter")
        cover_output = gr.Textbox(label="Cover Letter", lines=15)
        generate_btn.click(generate_cover_letter, inputs=[resume2, job2], outputs=cover_output)

demo.launch(share=True)