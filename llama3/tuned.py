import os  # os 모듈 운영체제와 상호 작용할 수 있는 기능을 제공
import torch  # PyTorch 라이브러리로, 주로 딥러닝과 머신러닝 모델을 구축, 학습, 테스트하는 데 사용
from datasets import load_dataset  # 데이터셋을 쉽게 불러오고 처리할 수 있는 기능을 제공
from transformers import (
    AutoModelForCausalLM,  # 인과적 언어 추론(예: GPT)을 위한 모델을 자동으로 불러오는 클래스
    AutoTokenizer,  # 입력 문장을 토큰 단위로 자동으로 잘라주는 역할
    BitsAndBytesConfig,  # 모델 구성
    HfArgumentParser,  # 파라미터 파싱
    TrainingArguments,  # 훈련 설정
    pipeline,  # 파이프라인 설정
    logging,  # 로깅을 위한 클래스
)

# 모델 튜닝을 위한 라이브러리
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

model_id = "../../models/Meta-Llama-3.1-8B"

dataset = load_dataset("Bingsu/ko_alpaca_data")

tokenizer = AutoTokenizer.from_pretrained("model_name")


def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(preprocess_function, batched=True)

model = AutoModelForCausalLM.from_pretrained("model_name")

training_args = TrainingArguments(
    output_dir="./results",  # 결과 저장 디렉토리
    evaluation_strategy="epoch",  # 평가 전략
    learning_rate=2e-5,  # 학습률
    per_device_train_batch_size=4,  # 배치 사이즈
    per_device_eval_batch_size=4,  # 평가 배치 사이즈
    num_train_epochs=3,  # 에폭 수
    weight_decay=0.01,  # 가중치 감쇠
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

trainer.evaluate()
