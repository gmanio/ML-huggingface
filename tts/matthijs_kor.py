import torch
from transformers import VitsTokenizer, VitsModel

mps_device = torch.device("mps")

model = VitsModel.from_pretrained("Matthijs/mms-tts-kor")
tokenizer = VitsTokenizer.from_pretrained("Matthijs/mms-tts-kor")

text = "안녕하세요. 한국어로 된 AI 보이스 입니다."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs)

# from IPython.display import Audio
# Audio(output.audio[0], rate=16000)