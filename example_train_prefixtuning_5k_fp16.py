from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaTokenizer
from peft import PrefixTuningConfig, get_peft_model, TaskType, get_peft_model_state_dict
from datasets import load_dataset
import sys

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)
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

# for Tesla V100
model = AutoModelForCausalLM.from_pretrained(
                        LLM_path, 
                        low_cpu_mem_usage=True, 
                        torch_dtype=torch.float16, 
                        device_map="auto", 
                )

prefix_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=10, 
            prefix_projection=False,
            )

model = get_peft_model(model, prefix_config)
model.print_trainable_parameters()
output_dir="./prefix_5k_fp16/checkpoint"
log_dir="./prefix_5k_fp16/log"
args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=1,
    logging_steps=10,
    logging_dir=log_dir,
    num_train_epochs=1,
    save_steps=2000,    
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val,
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

trainer.train()

trainer.model.save_pretrained("./prefix_5k_fp16/results")
tokenizer.save_pretrained("./prefix_5k_fp16/results")


