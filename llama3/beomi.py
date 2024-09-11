# beomi/Llama-3-Open-Ko-8B
import torch

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "../../models/Meta-Llama-3.1-8B"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     # 강의자료에서는 bfloat16을 이용하였으나
#     # 저는 auto 이용했습니다.
#     torch_dtype=torch.bfloat16,
#     # torch_dtype="auto",
#     device="mps",
# )

dataset = load_dataset("json", data_files="test.json", split="train")

sft_config = SFTConfig(
    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
    },
    max_seq_length=512,
    output_dir="/tmp",
    packing=False,
    num_train_epochs=4,
)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["prompt"])):
        text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts


trainer = SFTTrainer(
    model_id,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=formatting_prompts_func,
)

trainer.train()

trainer.model.save_pretrained("./dist/model")
trainer.tokenizer.save_pretrained("./dist/model")
