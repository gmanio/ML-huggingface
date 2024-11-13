import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "beomi/Llama-3-Open-Ko-8B"
model_id = "./test/model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # 강의자료에서는 bfloat16을 이용하였으나
    # 저는 auto 이용했습니다.
    torch_dtype=torch.bfloat16,
    # torch_dtype="auto",
    # device="cuda",
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
    max_new_tokens=256,
)

prompt = "벤탄쿠르의 근황은?"
# pipe = pipeline(
#     task="text-generation", model=model, tokenizer=tokenizer, max_length=200, text_inputs=prompt
# )

result = pipeline(f"### Question: {prompt}\n")
print(result)
