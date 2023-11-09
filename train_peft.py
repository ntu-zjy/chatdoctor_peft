import argparse
import torch
from functools import partial
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaTokenizer
from peft import get_peft_model, TaskType
from peft import LoraConfig
from peft import PrefixTuningConfig
from peft import PromptEncoderConfig, PromptEncoderReparameterizationType
from peft import PromptTuningConfig, PromptTuningInit

def load_data(data_path):
    ds = load_dataset("json", data_files=data_path)
    ds = ds['train']
    return ds

def load_tokenizer(tokenizer_path, load_local_tokenizer, tokenizer_save_directory=None):
    if load_local_tokenizer is True:
        assert tokenizer_save_directory is not None, "You have to set a directory if you want to load the tokenizer from your computer!"
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_save_directory)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        
    if tokenizer.pad_token is None:
        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "left"  
    return tokenizer

def process_func(example, tokenizer):
    MAX_LENGTH = 256
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

def set_peft_config(args):
    if args.peft_type=="prefix_tuning":
        config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=args.num_virtual_tokens, 
            prefix_projection=args.prefix_projections,
            )
    elif args.peft_type=="p_tuning":
        config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=args.num_virtual_tokens,
            encoder_reparameterization_type=args.encoder_reparameterization_type,
            encoder_dropout=args.encoder_dropout, 
            encoder_num_layers=args.encoder_num_layers, 
            encoder_hidden_size=args.encoder_hidden_size,
            )
    elif args.peft_type=="lora":
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
        )   
    elif args.peft_type=="prompt_tuning":
        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.num_virtual_tokens,
            tokenizer_name_or_path=args.tokenizer_path
            )
    return config

def set_trainer(tokenizer, model, train_data, args):
    checkpoint_dir = "./peft_weight/{}".format(args.peft_type)
    logging_dir = "./peft_weight/{}".format(args.peft_type)
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=args.save_step,
            output_dir=checkpoint_dir,
            save_total_limit=3,
            logging_dir=logging_dir,
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    return trainer

if __name__ == "__main__":
    # settings
    parser = argparse.ArgumentParser(description="train settings")
    # paths and base hyperparameters
    parser.add_argument("--model_name", type=str, default="default_model", help="??????AI????")
    parser.add_argument("--data_path", type=str, default="./data/HealthCareMagic-100k.json", help="??????")
    parser.add_argument("--tokenizer_path", type=str, default="linhvu/decapoda-research-llama-7b-hf", help="??????")
    parser.add_argument("--load_local_tokenizer", action="store_true", help="??????")
    parser.add_argument("--tokenizer_save_directory", type=str, default="./pretrained_weight/tokenizer_weight/", help="??????")
    parser.add_argument("--LLM_path", type=str, default="linhvu/decapoda-research-llama-7b-hf", help="??????")
    parser.add_argument("--load_local_LLM", action="store_true", help="??????")
    parser.add_argument("--LLM_save_directory", type=str, default="./pretrained_weight/LLM_weight", help="??????")
    parser.add_argument("--peft_type", type=str, default="lora", help="??????")
    parser.add_argument("--epochs", type=int, default=3, help="??????")
    parser.add_argument("--save_step", type=int, default=500, help="??????")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="??????")
    parser.add_argument("--batch_size", type=int, default=64, help="???????")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="???????")
    # peft settings
    parser.add_argument("--num_virtual_tokens", type=int, default=10, help="??????AI????")
    # prefix_tuning
    parser.add_argument("--prefix_projections", action="store_true", help="??????")
    # p_tuning
    parser.add_argument("--encoder_reparameterization_type", type=str, default=PromptEncoderReparameterizationType.MLP, help="??????AI????")
    parser.add_argument("--encoder_dropout", type=int, default=0.1, help="??????AI????")
    parser.add_argument("--encoder_num_layers", type=int, default=5, help="??????AI????")
    parser.add_argument("--encoder_hidden_size", type=int, default=1024, help="??????AI????")
    # lora
    parser.add_argument("--lora_target_modules", type=list, default=["q_proj", "k_proj"], help="??????AI????")
    parser.add_argument("--lora_r", type=int, default=8, help="??????AI????")
    parser.add_argument("--lora_alpha", type=int, default=16, help="??????AI????")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="??????AI????")
    parser.add_argument("--lora_bias", type=str, default="none", help="??????AI????")
    # prompt_tuning(use num_virtual_tokens and tokenizer_path)
    
    args = parser.parse_args()
    # load data
    ds = load_data(args.data_path)
    # tokenizer and data preprocess
    tokenizer = load_tokenizer(args.tokenizer_path, args.load_local_tokenizer, args.tokenizer_save_directory) 
    partial_process  = partial(process_func, tokenizer=tokenizer)
    tokenized_ds = ds.map(partial_process, remove_columns=ds.column_names)
    # set peft train config
    config = set_peft_config(args)
    # load base model
    if args.load_local_LLM is True:
        assert args.LLM_save_directory is not None, "You have to set a directory if you want to load the LLM from your computer!"
        model = AutoModelForCausalLM.from_pretrained(args.LLM_save_directory, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.LLM_path, low_cpu_mem_usage=True)
    # convert to a peft model
    model = get_peft_model(model, config)
    print("Peft Method:{}".format(args.peft_type))
    model.print_trainable_parameters()
    # set trainer
    trainer = set_trainer(tokenizer, model, tokenized_ds, args)
    model.config.use_cache = False
    # training!
    trainer.train()