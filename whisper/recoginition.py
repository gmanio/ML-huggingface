from transformers import pipeline

generator = pipeline(task="automatic-speech-recognition")
g = generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

print(g)
