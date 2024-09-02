import os

os.environ["CURL_CA_BUNDLE"] = ""

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7890"
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
QA_input = {
    "question": "Why is model conversion important?",
    "context": "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.",
}
res = nlp(QA_input)
