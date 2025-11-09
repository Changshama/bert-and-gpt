# DistilBERT vs OpenAI - NER Comparison App

A Gradio web application that compares Named Entity Recognition (NER) between a fine-tuned DistilBERT model and OpenAI's GPT model.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The OpenAI API key is already configured in the `.env` file.

## Run the App

```bash
python app.py
```

The Gradio interface will launch in your browser, typically at `http://127.0.0.1:7860`

## Features

- **Side-by-side comparison** of DistilBERT and OpenAI GPT for NER tasks
- **Pre-loaded examples** to test quickly
- **Live extraction** of named entities from any text input

## Models Used

- **DistilBERT**: `dslim/distilbert-NER` - Fine-tuned for token classification
- **OpenAI**: `gpt-4o-mini` - Cost-efficient GPT model

## Note

If you have a specific DistilBERT model fine-tuned on WNUT-17, you can replace the model name in `app.py:17` with your model path or HuggingFace model ID.
