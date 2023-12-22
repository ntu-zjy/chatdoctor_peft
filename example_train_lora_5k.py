from datasets import Dataset
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaTokenizer
from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict
from datasets import load_dataset
import torch
import sys

ds = load_dataset("json", data_files="./data/chatdoctor_test.json")
ds = ds['train']
print(ds)
tokenizer_path = "./pretrained_weight/tokenizer_weight"
LLM_path = "./pretrained_weight/LLM_weight"

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def process_func(example):
    MAX_LENGTH = 128
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Patient: " + example["instruction"], example["input"]]).strip() + "\n\nDoctor: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
model = AutoModelForCausalLM.from_pretrained(LLM_path, load_in_8bit=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)


target_modules = ["q_proj", "v_proj"]
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                         r=8, 
                         lora_alpha=16, 
                         target_modules=target_modules, 
                         lora_dropout=0.1, 
                         bias="none",
                         modules_to_save=["word_embeddings"])

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
output_dir="./lora/5k"
args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=3e-5,
    fp16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=100,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=2000,    
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

#trainer.train(resume_from_checkpoint=resume_from_checkpoint)
trainer.train()

trainer.model.save_pretrained("results")
tokenizer.save_pretrained("results")


