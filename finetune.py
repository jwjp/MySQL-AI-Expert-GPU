import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import os

# 1. 모델 및 토크나이저 설정
base_model_id = "beomi/Llama-3-Open-Ko-8B"
new_model_name = "mysql-expert-llama-3-8b" # 파인튜닝 후 저장될 모델 이름

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # 패딩 토큰 설정

# 2. 4비트 양자화 설정 (메모리 절약)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 3. 기반 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto" # GPU 자동 할당
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 4. LoRA 설정 (Parameter-Efficient Fine-Tuning)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. 데이터셋 로드
dataset = load_dataset("json", data_files="./data/finetune_data.jsonl", split="train")

# 6. 학습 인자(Arguments) 설정
training_arguments = TrainingArguments(
    output_dir="./results", # 학습 결과물 저장 경로
    num_train_epochs=1,     # 1 에포크만 학습 (데이터가 적으므로)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# 7. SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    # dataset_text_field="messages",
    # max_seq_length=None,
    # tokenizer=tokenizer,
    args=training_arguments,
    # packing=False,
)

# 8. 학습 시작
print("Fine-tuning을 시작합니다.")
trainer.train()
print("Fine-tuning 완료!")

# 9. 학습된 모델(어댑터) 저장
trainer.model.save_pretrained(new_model_name)
print(f"학습된 모델이 '{new_model_name}' 경로에 저장되었습니다.")

# (선택사항) 기반 모델과 어댑터 병합 후 저장
# base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, return_dict=True, device_map='auto')
# merged_model = PeftModel.from_pretrained(base_model, new_model_name)
# merged_model = merged_model.merge_and_unload()
# merged_model.save_pretrained("merged_model", safe_serialization=True)
# tokenizer.save_pretrained("merged_model")


