import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from transformers import pipeline

model_id = "beomi/Llama-3-Open-Ko-8B"
peft_model_id = "/home/caidev/workspace/llmops/mjchoi/devocean-llm/llama3-finetuning/autotrain/model"

config = PeftConfig.from_pretrained(peft_model_id)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.eval()

def gen(prompt):
    prompt = "알바생이 3일 일하고 그만뒀는데 주휴수당을 줘야 하나요?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

gen("알바생이 3일 일하고 그만뒀는데 주휴수당을 줘야 하나요?")