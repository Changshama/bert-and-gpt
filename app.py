import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load DistilBERT model fine-tuned on WNUT-17
# Using a pre-trained model on WNUT-17 dataset
model_name = "dslim/distilbert-NER"  # This is fine-tuned on CoNLL-2003, but similar task
# If you have a specific WNUT-17 model, replace with: "your-username/distilbert-wnut17"

print("Loading DistilBERT model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
print("DistilBERT model loaded!")

def extract_entities_distilbert(text):
    """Extract entities using fine-tuned DistilBERT model."""
    try:
        entities = ner_pipeline(text)

        # Format the results
        if not entities:
            return "No entities found."

        result = "**Entities Found:**\n\n"
        for entity in entities:
            result += f"- **{entity['word']}** ({entity['entity_group']}): {entity['score']:.3f}\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"

def extract_entities_openai(text):
    """Extract entities using OpenAI GPT model."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o-mini for cost efficiency
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at Named Entity Recognition. Extract all named entities from the given text and categorize them into one of the categories location (LOC), organizations (ORG), person (PER), and Miscellaneous (MISC). Format your response as a list with entity name, type, and probability."
                },
                {
                    "role": "user",
                    "content": f"Extract named entities from this text:\n\n{text}"
                }
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def compare_models(text):
    """Compare both models on the same input."""
    if not text.strip():
        return "Please enter some text.", "Please enter some text."

    distilbert_result = extract_entities_distilbert(text)
    openai_result = extract_entities_openai(text)

    return distilbert_result, openai_result

# Create Gradio interface
with gr.Blocks(title="DistilBERT vs OpenAI - NER Comparison") as demo:
    gr.Markdown(
        """
        # Named Entity Recognition: DistilBERT vs OpenAI

        Compare a fine-tuned DistilBERT model with OpenAI's GPT model for Named Entity Recognition tasks.

        **DistilBERT**: Fast, efficient, runs locally
        **OpenAI GPT**: Powerful, context-aware, requires API calls
        """
    )

    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text",
            placeholder="Enter text to extract named entities...",
            lines=5,
            value="Apple Inc. announced a new product in San Francisco. Tim Cook presented the new iPhone."
        )

    submit_btn = gr.Button("Extract Entities", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### DistilBERT Results")
            distilbert_output = gr.Markdown(label="DistilBERT Output")

        with gr.Column():
            gr.Markdown("### OpenAI GPT Results")
            openai_output = gr.Markdown(label="OpenAI Output")

    submit_btn.click(
        fn=compare_models,
        inputs=[input_text],
        outputs=[distilbert_output, openai_output]
    )

    # Add example inputs
    gr.Examples(
        examples=[
            # Easy examples
            ["Apple Inc. announced a new product in San Francisco. Tim Cook presented the new iPhone."],
            ["The United Nations held a meeting in New York to discuss climate change with representatives from China and Germany."],
            ["Elon Musk's company SpaceX launched a rocket from Cape Canaveral yesterday."],
            ["The Beatles were a famous band from Liverpool, England."],

            # Challenging examples
            ["I bought an Apple from the store, then went to Apple to buy a new iPhone."],
            ["Anthropic released Claude 3.5 Sonnet last year, competing with OpenAI's GPT-4o and Google's Gemini."],
            ["The University of California, San Francisco Medical Center announced a partnership with the Bill & Melinda Gates Foundation."],
            ["elon musk tweeted about spacex's starship launch from boca chica, texas yesterday."],
            ["COVID-19 spread from Wuhan in December 2019, affecting WHO's response protocol."],
            ["The CEO of IBM, formerly with GE and MIT, spoke at the UN about AI and ML regulations."],
            ["Tim Cook met with Harry Potter author J.K. Rowling at Hogwarts... actually at Edinburgh Castle in Scotland."],
        ],
        inputs=[input_text],
    )

if __name__ == "__main__":
    print("Starting Gradio app...")
    demo.launch()
