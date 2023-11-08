from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.io.wavfile import write

processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")

inputs = processor(
    text=["electric feelings with daftpunk beat"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=2048)

sampling_rate = model.config.audio_encoder.sampling_rate
write("musicgen_out.wav", rate=sampling_rate, data=audio_values.cpu().numpy())