from transformers import VitsModel, AutoTokenizer
import torch
from scipy.io.wavfile import write
import numpy as np

model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")

text = "Annyŏnghaseyo. pan'gapsŭmnida."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform
    
# scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)
# waveform = output.waveform[0]

# write("test.wav", rate=model.config.sampling_rate, data=output)
write("example.wav", rate=model.config.sampling_rate, data=output.cpu().numpy().squeeze())