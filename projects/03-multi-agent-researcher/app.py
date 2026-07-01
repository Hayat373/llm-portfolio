import gradio as gr
from agents.coordinator import Coordinator

coordinator = Coordinator()

def multi_agent_chat(message, history):
    response = coordinator.process_query(message)
    return response

demo = gr.ChatInterface(
    fn=multi_agent_chat,
    title="🤖 Multi-Agent Research Assistant",
    description="Researcher + Writer Agents working together • Ask complex questions about Ethiopia or general topics",
    examples=[
        "What are the main tourist attractions in Ethiopia?",
        "Tell me about the history and culture of Ethiopia.",
        "What should I know before visiting Addis Ababa?"
    ]
)

demo.launch(share=True)