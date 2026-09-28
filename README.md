# T5 Text Summarization App

A simple text summarization project built as a Transformer-based NLP project using T5, Hugging Face, and FastAPI.

## About the Project

The main idea of this project is to take a conversation or piece of text and generate a shorter summary while keeping the important information.

I used the T5-small Transformer model and fine-tuned it for dialogue summarization using the SAMSum dataset.

The trained model is then connected to a FastAPI backend and a simple web interface where users can enter text and get the generated summary.

## What I Used

- Python
- PyTorch
- Hugging Face Transformers
- T5-small
- FastAPI
- Jinja2
- HTML, CSS and JavaScript
- Uvicorn
- SAMSum Dataset

## How the Project Works

```text
User enters text
       ↓
FastAPI receives the text
       ↓
Text is cleaned and tokenized
       ↓
T5 model generates the summary
       ↓
Summary is shown on the webpage