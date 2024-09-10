from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer

# dataset = load_dataset("imdb", split="train")
dataset = load_dataset("json", data_files="test.json", split="train")

print(dataset)
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["prompt"])):
        text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts


sft_config = SFTConfig(
    model_init_kwargs={
        "torch_dtype": "bfloat16",
    },
    # max_seq_length=256,
    output_dir="./dist",
    packing=False,
    num_train_epochs=10
)

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=sft_config,
    formatting_func=formatting_prompts_func,
    
)

trainer.train()

trainer.model.save_pretrained("./dist/model")
trainer.tokenizer.save_pretrained("./dist/model")