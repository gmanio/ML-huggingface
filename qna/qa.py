import os

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7890"

from transformers import pipeline

model_name = "deepset/roberta-base-squad2"

qa_model = pipeline("question-answering", model=model_name, tokenizer=model_name)
question = "Where do I live?"
context = "My name is Merve and I live in İstanbul."
qa_model(question=question, context=context)
