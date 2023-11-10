from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaTokenizer

model = AutoModelForCausalLM.from_pretrained("linhvu/decapoda-research-llama-7b-hf", low_cpu_mem_usage=True)

save_directory = "./chatdoctor_base/pretrained_weight"
model.save_pretrained(save_directory)



tokenizer_save_directory = "./chatdoctor_base/pretrained_weight"
tokenizer = LlamaTokenizer.from_pretrained("linhvu/decapoda-research-llama-7b-hf")
tokenizer.save_pretrained(tokenizer_save_directory)
