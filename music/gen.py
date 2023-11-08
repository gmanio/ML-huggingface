from transformers import pipeline

from scipy.io.wavfile import write
synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True})

write("musicgen_out.wav", rate=music["sampling_rate"], music=music["audio"].cpu().numpy().squeeze())