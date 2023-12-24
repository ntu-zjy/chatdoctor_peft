from transformers import LlamaForCausalLM, LlamaTokenizer
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
save_directory = "./pretrained_weight"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

#from modelscope.hub.snapshot_download import snapshot_download

#model_dir = snapshot_download('skyline2006/llama-7b', cache_dir=save_directory)

