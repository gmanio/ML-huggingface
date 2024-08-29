from diffusers import StableDiffusion3Pipeline
import torch
import asyncio
import random

prompts = [
    "A 3D rendering of a tree with bright yellow leaves and an abstract style.",
    "An illustration of a mountain in the style of Impressionism with a wide aspect ratio.",
    "A photograph of a steampunk alien taken from a low-angle viewpoint.",
    "A sketch of a raccoon in bright colors and minimalist composition.",
    "A 3D rendering of a spaceship in the style of cubism with high resolution.",
    "An old woman washing her clothes in the winter in the style of Renaissance art with a narrow aspect ratio.",
    "A 3D rendering of a star with pastel colors and a whimsical look.",
    "A picture of a butterfly riding a motorcycle in vaporware style with a wide-angle view.",
    "A photograph of three friends playing music on the street in the style of Pop Art with a medium aspect ratio.",
    "A group of pug dogs at a rave in a Renaissance style.",
    "A sketch of a mysterious castle in the style of Gothic art with an aerial viewpoint.",
    "A 3D rendering of an office desk with a futuristic look and bokeh.",
    "Cubist painting of a backlit train station with bright colors and realistic textures.",
    "An illustration of a woman laying on a bed in a dynamic pose dreaming in black-and-white.",
    "Impressionist oil painting of a beach at sunset with a narrow aspect ratio.",
    "A photograph of a city skyline in the style of Edward Hopper taken from an aerial viewpoint.",
    "A 3D rendering of a cat sitting on a windowsill in minimalist style with high resolution.",
    "Graffiti-style painting of a city street with an urban look and textured surfaces.",
    "A sketch of a pirate ship in black-and-white with realistic textures and low resolution.",
    "A chalk drawing of a family picnic being attacked by ants in Central Park with a surrealist style.",
    "A watercolor painting of a coffee shop with surreal elements in vibrant colors.",
    "An oil painting of a rainbow over a rural abandoned town with classic style.",
    "A 3D rendering of a spaceship taking off into space with a cyberpunk look and wide aspect ratio.",
    "A futuristic space station is shown in jewel tones, ultra photoreal, and cinematic lighting.",
    "A creative composition of a frog wearing a crown sitting on a log in a Japanese anime style.",
    "Outside of a 1960s diner in monochromatic colors and vintage feel.",
    "A retro-style robot playing a futuristic video game in neon tones with medium resolution.",
    "An image of a dark and mysterious castle with bats flying around it in an American Gothic style.",
    "An animated GIF of a robot dancing to 80s music with a cartoon look.",
    "An illustration of a family photo taken on the beach in the style of Ansel Adams.",
    "An abstract painting of a sunflower in the middle of a desert with bright colors and low resolution.",
    "A beachfront bar at a holiday resort at nighttime with purple and pink tones in a pop art style.",
    "A 3D rendering of a garden blooming with flowers under the moonlight in a low-angle view.",
    "A fantasy painting of a castle sitting on top of a craggy peak with a cinematic tone.",
    "A candid photograph of a woman standing at the edge of a cliff overlooking the ocean in an art nouveau style.",
    "A 3D rendering of a cityscape at night with neon lights and an abstract style.",
    "An illustration of a spaceship flying through the stars in the style of Van Gogh with a full depth of field.",
    "A sketch of two cats sitting on a sofa watching TV while eating spaghetti.",
    "Two monsters playing chess in the style of cubism with bright colors and low resolution.",
    "A picture of a person walking alone through a forest in the style of Romanticism taken from an aerial viewpoint.",
    "A low-detail oil painting of a thunderstorm over a cityscape with dark tones and a backlit resolution.",
    "A 3D rendering of a futuristic train station in the style of Art Nouveau with volumetric lighting.",
    "An illustration of a river winding through a meadow in the style of Impressionism with a thick black outline.",
    "A photograph of a person sitting on a bench facing the sunset in black and white.",
    "A minimalist painting of a city skyline in bright colors and high resolution.",
    "A sketch of two robots talking to each other with a surreal look and narrow aspect ratio.",
    "A Dadaist collage of a post-apocalyptic world in neon tones and 4K resolution.",
    "A rococo painting of a garden with abstract elements and high resolution.",
    "A photograph of an old man walking in the rain making eye contact with the viewer in a mid-shot view.",
    "A watercolor painting of a flock of birds flying over a river at sunset with realistic textures.",
]

mps_device = torch.device("mps")

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16,
)
pipe = pipe.to(mps_device)


async def do_work(index, propmt):
    # randomNumber = random.randint(0, 10)
    print(f"{propmt, index} 시작")
    # await asyncio.sleep(randomNumber)
    with torch.no_grad():
        image = pipe(
            propmt,
            negative_prompt="",
            num_inference_steps=20,
            guidance_scale=7.0,
            height=1024,
            width=1024,
        ).images[0]
        image.save("result" + str(index) + ".png")
    print(f"{propmt, index} 완료")


async def sequential_loop():
    # 작업을 하나씩 차례로 실행합니다.
    # for name, delay in tasks:
    #     await do_work(name, delay)

    for index, propmt in enumerate(prompts):
        await do_work(index, propmt)


# 비동기 루프를 실행합니다.
if __name__ == "__main__":
    asyncio.run(sequential_loop())
