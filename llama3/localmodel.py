import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "meta-llama/Meta-Llama-3.1-8B"
model_id = "./dist/model"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda"
# )

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda", tokenizer=tokenizer
)

prompt= "what age?"
result = pipeline(f"### Question: {prompt}\n")

print(result)