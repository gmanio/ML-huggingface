import torch

from transformers import pipeline

from scipy.io.wavfile import write

mps_device = torch.device("mps")

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True})

# write("musicgen_out.wav", rate=music["sampling_rate"], music=music["audio"].cpu().numpy().squeeze())
write("musicgen_out.wav", rate=music["sampling_rate"], music=music["audio"].to().numpy().squeeze())