import transformers
import torch

# model_id = "beomi/Llama-3-Open-Ko-8B"
model_id = "../../models/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="mps",
    max_new_tokens=2048,
)

# messages = [
#     {
#         "role": "system",
#         "content": "You are a pirate chatbot who always responds in pirate speak!",
#     },
#     {"role": "user", "content": "Who are you?"},
# ]

# outputs = pipeline(
#     messages,
#     max_new_tokens=256,
# )

result = pipeline("사자, 힘, 용기라는 단어로 문장 만들기")

print(result)
