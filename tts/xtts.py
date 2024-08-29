import torch
from TTS.api import TTS

mps_device = torch.device("mps")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
# generate speech by cloning a voice using default settings
tts.tts_to_file(text="와 이게 아이유 목소리로 한글로 잘 번역이 되네",
                file_path="output.wav",
                speaker_wav="./iu.mp3",
                language="ko")