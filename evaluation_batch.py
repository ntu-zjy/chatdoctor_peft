from datasets import Dataset
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaTokenizer, LlamaForCausalLM
from peft import PrefixTuningConfig, PromptEncoderConfig, PeftConfig, get_peft_model, TaskType, PeftModel, PromptEncoderReparameterizationType, PromptTuningConfig, PromptTuningInit
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import os
import sys

@torch.no_grad()
def go(pmsg, instruction):
    invitation = "Doctor: "
    human_invitation = "Patient: "

    # input
    #msg = input(human_invitation)
    msg = human_invitation+pmsg
    
    fulltext = instruction + " \n\n" + msg + "\n\n" + invitation
    #print('SENDING==========')
    #print(fulltext)
    #print('==========')

    generated_text = ""
    #print("fulltext:", fulltext)
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
    ##print("gen_in:", gen_in)
    in_tokens = len(gen_in)
    generated_ids = peft_model.generate(
        input_ids=gen_in,
        max_new_tokens=256,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
        temperature=0.5, # default: 1.0
        top_k = 50, # default: 50
        top_p = 0.9, # default: 1.0
        #early_stopping=False,
    )
    #print("generated_ids:", generated_ids)
    generated_text = tokenizer.batch_decode(generated_ids.detach().cpu().numpy(), skip_special_tokens=True)
    generated_text = generated_text[0]
    #print("generated_text:", generated_text)

    text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt
    #print("response:", response)
    response = response.split(human_invitation)[0]
    #print("response:", response)   
    response.strip()
    
    return response


def read_json(path):
    # Reading data
    with open(path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    # give sys parameters
    if len(sys.argv) > 1:
        name_of_method = sys.argv[1]

    LLM_path = "./pretrained_weight"
    Peft_path = "./{}/results".format(name_of_method)
    tokenizer_path = Peft_path
    data_path = "./data/chatdoctor5k_test.json"
    result_data_path = "./data/chatdoctor5k_eval_"+name_of_method+".json"

    base_model = LlamaForCausalLM.from_pretrained(LLM_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    peft_config = PeftConfig.from_pretrained(Peft_path)
    peft_model = PeftModel.from_pretrained(model=base_model, model_id=Peft_path).cuda()
    peft_model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    data = read_json(data_path)
    data = data
    eval_result = []
    print('testing...')
    for d in tqdm(data):
        pmsg =  d["input"] 
        instruction = d["instruction"]
        output = go(pmsg, instruction)
        eval_result.append({"instruction":instruction, "input":pmsg, "output":output})
        print({"instruction":instruction, "input":pmsg, "output":output})
    print('write json...')
    with open(result_data_path, 'w+') as f:
        json.dump(eval_result, f, indent=2)
    print('finished')