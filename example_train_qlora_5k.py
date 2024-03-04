from datasets import Dataset
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaTokenizer
from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict
from datasets import load_dataset
import torch
import sys

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)
ds = load_dataset("json", data_files="./data/HealthCareMagic-100k.json")
#ds = load_dataset("json", data_files="./data/chatdoctor5k.json")
ds = ds['train']
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
    full_prompt = tokenizer(generate_prompt(example), add_special_tokens=False)
    #print(generate_prompt(example))
    #response = tokenizer(example["output"], add_special_tokens=False)
    input_ids = full_prompt["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = full_prompt["attention_mask"] + [1]
    labels = full_prompt["input_ids"] + [tokenizer.eos_token_id]
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

model = AutoModelForCausalLM.from_pretrained(
                        LLM_path, 
                        torch_dtype=torch.float16, 
                        device_map="auto", 
                        load_in_4bit=True, 
                        bnb_4bit_compute_dtype=torch.half,
                        bnb_4bit_quant_type="nf4", 
                        bnb_4bit_use_double_quant=True,
                )
#
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                         r=8, 
                         lora_alpha=16,
                         target_modules = ["q_proj", "v_proj"],
                         lora_dropout=0.05, 
                         bias="none",
                         )

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
output_dir="./qlora/checkpoint"
log_dir="./qlora/log"
args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=3e-5,
    per_device_train_batch_size=6, 
    gradient_accumulation_steps=1,
    logging_steps=10,
    logging_dir=log_dir,
    num_train_epochs=3,
    optim="paged_adamw_32bit",
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
trainer.train()

trainer.model.save_pretrained("./qlora/results")
tokenizer.save_pretrained("./qlora/results")


