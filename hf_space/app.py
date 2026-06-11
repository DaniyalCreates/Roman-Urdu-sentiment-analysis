import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "DaniyalCreates/roman-urdu-sentiment-xlmroberta"

print("Loading model…")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
_model.eval()
print("Model ready.")


def predict(text: str):
    if not text or not text.strip():
        return {}
    inputs = _tokenizer(
        text.strip(), return_tensors="pt", truncation=True, max_length=128
    )
    with torch.no_grad():
        probs = torch.softmax(_model(**inputs).logits, dim=1).squeeze()
    pos, neg = probs[1].item(), probs[0].item()
    return {"Positive 😊": pos, "Negative 😞": neg}


EXAMPLES = [
    ["yaar ye movie bohat achi thi"],
    ["ye service bilkul bakwas hai"],
    ["mujhe ye phone pasand nahi aya"],
    ["bohat maza aya aaj"],
    ["ye banda bilkul jhota hai"],
]

with gr.Blocks(
    title="Roman Urdu Sentiment Analysis",
    theme=gr.themes.Soft(),
    css="""
    .contain { max-width: 680px; margin: auto; }
    #title { text-align: center; margin-bottom: 4px; }
    #subtitle { text-align: center; color: #6b7280; margin-bottom: 24px; font-size: 0.95rem; }
    """,
) as demo:
    with gr.Column(elem_classes=["contain"]):
        gr.Markdown("# Roman Urdu Sentiment Analysis", elem_id="title")
        gr.Markdown(
            "Type a sentence in Roman Urdu — the model predicts **Positive** or **Negative** "
            "and shows confidence for both classes.",
            elem_id="subtitle",
        )

        txt = gr.Textbox(
            label="Roman Urdu Text",
            placeholder="e.g. yaar ye movie bohat achi thi…",
            lines=3,
        )
        btn = gr.Button("Analyse", variant="primary", size="lg")
        out = gr.Label(label="Sentiment", num_top_classes=2)

        gr.Examples(examples=EXAMPLES, inputs=txt, label="Try an example")

    btn.click(fn=predict, inputs=txt, outputs=out)
    txt.submit(fn=predict, inputs=txt, outputs=out)

demo.launch()
