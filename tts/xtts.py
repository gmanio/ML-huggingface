import torch
from TTS.api import TTS

mps_device = torch.device("mps")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
# generate speech by cloning a voice using default settings
tts.tts_to_file(text="야 저리가 내 자리에 두번다시 오지마",
                file_path="output.wav",
                speaker_wav="./female.wav",
                language="ko")