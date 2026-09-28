#pip install fastapi uvicorn

# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import torch
# import re
# from fastapi.templating import Jinja2Templates  # ui
# from fastapi.responses import HTMLResponse  
# from fastapi.staticfiles import StaticFiles  

# # initialize our fast api app
# app = FastAPI(
#     title="Text Summarizer App",
#     description="Text Summarizer using T5",
#     version="1.0"
# )

# # load the tokenizer and model
# model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
# tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

#  # device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # templating
# templates = Jinja2Templates(directory=".")

# # Input Schema (Dialouge -> format)
# class DialougeInput(BaseModel):
#     dialouge: str

# def clean_data(text):
#     text = re.sub(r"\r\n", " ", text) # lines
#     text = re.sub(r"<.*?>", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     text = text.strip().lower()

#     return text

# def summarize_dilaouge(dialogue : str) -> str:
#   dialogue = clean_data(dialogue)

#   # tokenize
#   inputs = tokenizer(dialogue, padding = "max_length", return_tensors = "pt", max_length = 512, truncation = True).to(device)


#   # generate summary -> token ids

#   model.to(device)
#   targets = model.generate(
#       input_ids = inputs["input_ids"],
#       attention_mask = inputs["attention_mask"],
#       max_length = 150,
#       num_beams = 4,          # 4 diff O/Ps or summary sequences
#       early_stopping = True
#       # repetition_penalty = 3.0
#   )

#   # convert token ids to summary -> decoded our O/Ps

#   summary = tokenizer.decode(
#       targets[0],
#       skip_special_tokens = True,    # EOS, SEP
#       # clean_up_tokenization_spaces = True
#   )
#   return summary


# # IMP -> API endpoint
# @app.post("/summarized/")
# async def summarize(dialouge_input: DialougeInput):
#     summary = summarize_dilaouge(dialouge_input.dialouge)
#     return {"summary": summary}  # JSON Obj return

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse(
#     request=request,
#     name="index.html"
# )


from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarizer using T5",
    version="1.0"
)

# Load tokenizer and model
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Templating
templates = Jinja2Templates(directory=".")


# Input Schema
class DialougeInput(BaseModel):
    dialouge: str


# Clean text
def clean_data(text):
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    return text


# Summarize dialogue
def summarize_dilaouge(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    # Tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    # Generate summary
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # Convert token IDs to text
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# API endpoint
@app.post("/summarized/")
async def summarize(dialouge_input: DialougeInput):
    summary = summarize_dilaouge(dialouge_input.dialouge)
    return {"summary": summary}


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
