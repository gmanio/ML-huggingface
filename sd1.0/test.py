from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32)
# pipe.to("cuda")
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.enable_model_cpu_offload()
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "A vintage travel poster for Venus in portrait orientation. The scene portrays the thick, yellowish clouds of Venus with a silhouette of a vintage rocket ship approaching. Mysterious shapes hint at mountains and valleys below the clouds. The bottom text reads, 'Explore Venus: Beauty Behind the Mist'. The color scheme consists of golds, yellows, and soft oranges, evoking a sense of wonder."

images = pipe(prompt=prompt).images[0]

images.save("test1.png")