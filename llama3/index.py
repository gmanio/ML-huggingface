import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model_id = "meta-llama/Meta-Llama-3.1-8B"
model_id = "../../models/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # 강의자료에서는 bfloat16을 이용하였으나
    # 저는 auto 이용했습니다.
    torch_dtype=torch.bfloat16,
    # torch_dtype="auto",
    device="mps",
)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="mps",
# )
outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    # beomi 모델의 경우 temperature 를 1로 줌 (더 다양한 답변 생성)
    # temperature=1,
    temperature=0.6,
    top_p=0.9,
)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])
