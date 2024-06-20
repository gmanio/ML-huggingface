import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

mps_device = torch.device("mps")

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"], device=mps_device).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("This model is part of Facebook's Massively Multilingual Speech project, aiming to provide speech technology across a diverse range of languages. You can find more details about the supported languages and their ISO 639-3 codes in the MMS Language Coverage Overview.", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])