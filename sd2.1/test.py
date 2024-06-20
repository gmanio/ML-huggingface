from diffusers import StableDiffusionPipeline
import torch

mps_device = torch.device("mps")

print(torch.backends.mps.is_available())

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(mps_device)

prompt = "an illustrate of macbook on the earth."
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")