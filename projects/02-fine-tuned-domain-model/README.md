# 02. Fine-tuned Ethiopian Domain Model

**LoRA Fine-tuning on TinyLlama**  
A customized LLM specialized in Ethiopian knowledge, culture, history, and local context.

### Features
- Fine-tuned using **LoRA** (efficient fine-tuning)
- 4-bit quantized for fast inference
- Domain-specific knowledge about Ethiopia
- Before vs After comparison available

### Tech Stack
- Base Model: TinyLlama-1.1B
- Fine-tuning: PEFT (LoRA) + bitsandbytes 4-bit
- Evaluation: Custom Ethiopian dataset

### Results
- Improved responses on Ethiopian topics
- Maintains general knowledge while excelling in domain-specific questions

### How to Use
1. Open `inference.ipynb` in Colab
2. Run the cells to load the fine-tuned model
3. Chat with the Ethiopian-specialized AI

### Future Improvements
- Larger dataset
- Merge with other domain adapters
- Deploy as a web app

