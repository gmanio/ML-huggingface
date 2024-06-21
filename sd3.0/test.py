from diffusers import StableDiffusion3Pipeline
import torch

mps_device = torch.device("mps")

print(torch.backends.mps.is_available())

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16)
pipe = pipe.to(mps_device)

image = pipe(
    "a super car looks like lamborghini, purple color, nice mupler, great tire with black wheel",
    negative_prompt="",
    num_inference_steps=20,
    guidance_scale=7.0,
    height=1024,
    width=1024,
).images[0]
    
image.save("car01.png")