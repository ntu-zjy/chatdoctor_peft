from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaTokenizer, AutoTokenizer
from peft import PromptEncoderConfig, get_peft_model, TaskType, PromptEncoderReparameterizationType
from datasets import load_dataset

#ds = load_dataset("json", data_files="./data/HealthCareMagic-100k.json")
ds_train = load_dataset("json", data_files="./data/chatdoctor5k_train.json")
ds_train = ds_train['train']
ds_val = load_dataset("json", data_files="./data/chatdoctor5k_val.json")
ds_val = ds_val['train']
tokenizer_path = "./pretrained_weight"
LLM_path = "./pretrained_weight"

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def generate_prompt(example):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {example["instruction"]}
                
                ### Input:
                {example["input"]}
                
                ### Response:
                {example["output"]}"""

def generate_prompt_old(example):
    return f"""Below is an instruction that describes a task.

                ### Instruction:
                {example["instruction"]}
                
                ### Patient:
                {example["input"]}
                
                ### Doctor:
                {example["output"]}"""

def process_func(example):
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []
    full_prompt = tokenizer(generate_prompt(example))
    user_prompt = tokenizer(generate_prompt({**example, "output":""}))
    user_prompt_length = len(user_prompt["input_ids"])
    input_ids = full_prompt["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = full_prompt["attention_mask"] + [1]
    labels = [-100] * user_prompt_length + full_prompt["input_ids"][user_prompt_length:] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds_train = ds_train.map(process_func, remove_columns=ds_train.column_names)
tokenized_ds_val = ds_val.map(process_func, remove_columns=ds_val.column_names)

# check data processing
print(tokenizer.decode(tokenized_ds_train[0]["input_ids"]))
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds_train[0]["labels"]))))

# for Tesla V100
model = AutoModelForCausalLM.from_pretrained(
                        LLM_path, 
                        low_cpu_mem_usage=True, 
                        torch_dtype=torch.float16, 
                        device_map="auto", 
                )

ptuning_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=20,
            encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
            encoder_dropout=0.1, 
            )

model = get_peft_model(model, ptuning_config)
model.print_trainable_parameters()
output_dir="./ptuning_5k_fp16/checkpoint"
log_dir="./ptuning_5k_fp16/log"
args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=3e-5,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=2,
    logging_steps=10,
    logging_dir=log_dir,
    num_train_epochs=3,
    save_steps=2000,    
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

trainer.model.save_pretrained("./ptuning_5k_fp16/results")
tokenizer.save_pretrained("./ptuning_5k_fp16/results")


