from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaTokenizer
#model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-chat", low_cpu_mem_usage=True)

save_directory = "./pretrained_weight"

#model.save_pretrained(save_directory)

#tokenizer_save_directory = "./pretrained_weight"
#tokenizer = LlamaTokenizer.from_pretrained("sharpbai/Llama-2-7b-chat")
#tokenizer.save_pretrained(tokenizer_save_directory)
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('skyline2006/llama-7b', cache_dir=save_directory)

