from datasets import Dataset
<<<<<<< HEAD
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaForCausalLM, LlamaTokenizer
=======
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, LlamaTokenizer
>>>>>>> 57d2455b820e4d5d58e6ca9c21f0986fcc540529
from peft import PrefixTuningConfig, PromptEncoderConfig, PeftConfig, get_peft_model, TaskType, PeftModel, PromptEncoderReparameterizationType, PromptTuningConfig, PromptTuningInit
from datasets import load_dataset
import torch

<<<<<<< HEAD
tokenizer_path = "./pretrained_weight"
LLM_path = "./pretrained_weight"
Peft_path = "./ptuning_5k_fp16/results"
tokenizer_path = Peft_path
#Peft_path = "./lora/5k/latest/"
'''
base_model = LlamaForCausalLM.from_pretrained(
                        LLM_path, 
                        torch_dtype=torch.half, 
                        device_map="auto", 
                        load_in_4bit=True, 
                        bnb_4bit_compute_dtype=torch.half,
                        bnb_4bit_quant_type="nf4", 
                        bnb_4bit_use_double_quant=True,
                )
'''
#base_model = AutoModelForCausalLM.from_pretrained(LLM_path, load_in_8bit=True, low_cpu_mem_usage=True)
base_model = AutoModelForCausalLM.from_pretrained(LLM_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
peft_config = PeftConfig.from_pretrained(Peft_path)
peft_model = PeftModel.from_pretrained(model=base_model, model_id=Peft_path).cuda()

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
#tokenizer = LlamaTokenizer.from_pretrained(LLM_path)
peft_model.eval()
generator = peft_model.generate
#tokenizer = LlamaTokenizer.from_pretrained(LLM_path)
#generator = base_model.generate

def go():
    invitation = "Doctor: "
=======
tokenizer_path = "./pretrained_weight/tokenizer_weight"
LLM_path = "./pretrained_weight/LLM_weight"
Peft_path = "checkpoint-221500"

config = PeftConfig.from_pretrained(Peft_path)

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, low_cpu_mem_usage=True)

peft_model = PeftModel.from_pretrained(model=model, model_id=Peft_path)
peft_model = peft_model.cuda()

tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
generator = peft_model.generate

'''
fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + "Patient: {}\n{}".format("Doctor, I have been having trouble with my eye alignment lately. I often see double and feel eye strain, especially when I read or look at something for a long time.", "").strip() + "\n\nDoctor: "
ipt = tokenizer(fulltext, return_tensors="pt").to(peft_model.device)
generator = model.generate
with torch.no_grad():
    generated_ids = generator(
        **ipt,
        max_new_tokens=200,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
        temperature=0.5, # default: 1.0
        top_k = 50, # default: 50
        top_p = 1.0, # default: 1.0
        early_stopping=True,
    )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

    text_without_prompt = generated_text[len(fulltext):]
print(text_without_prompt)

with torch.no_grad():
    
    print(tokenizer.decode(peft_model.generate(ipt, max_new_tokens=200, do_sample=True)[0], skip_special_tokens=True))
'''
def go():
    history = []
    invitation = "ChatDoctor: "
>>>>>>> 57d2455b820e4d5d58e6ca9c21f0986fcc540529
    human_invitation = "Patient: "

    # input
    #msg = input(human_invitation)
    msg = human_invitation+"Hello doctor, I have been going to a dentist about a tooth that is sore. They started a root canal but then did not finish. I have gone back several times and every time they open up the tooth and insert medicine. My tooth is in agony right now. I am just wondering if this is a standard procedure during a root canal? Should not the dentist remove the nerves so I am not in so much pain? Or is it the standard procedure to fight the infection first and then remove the nerves?"
    print("")

<<<<<<< HEAD
    fulltext = msg + "\n\n" + invitation
    #fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + msg + "\n\n" + invitation
=======
    history.append(human_invitation + msg)

    fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + "\n\n".join(history) + "\n\n" + invitation
>>>>>>> 57d2455b820e4d5d58e6ca9c21f0986fcc540529
    #fulltext = "\n\n".join(history) + "\n\n" + invitation
    
    #print('SENDING==========')
    #print(fulltext)
    #print('==========')

    generated_text = ""
<<<<<<< HEAD
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
=======
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.to(peft_model.device)
>>>>>>> 57d2455b820e4d5d58e6ca9c21f0986fcc540529
    in_tokens = len(gen_in)
    with torch.no_grad():
            generated_ids = generator(
                input_ids=gen_in,
<<<<<<< HEAD
                max_new_tokens=1000,
=======
                max_new_tokens=200,
>>>>>>> 57d2455b820e4d5d58e6ca9c21f0986fcc540529
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5, # default: 1.0
                top_k = 50, # default: 50
                top_p = 1.0, # default: 1.0
<<<<<<< HEAD
                early_stopping=False,
=======
                early_stopping=True,
>>>>>>> 57d2455b820e4d5d58e6ca9c21f0986fcc540529
            )
            generated_text = tokenizer.batch_decode(generated_ids.detach().cpu().numpy(), skip_special_tokens=True)[0] 

            text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt

    response = response.split(human_invitation)[0]

    response.strip()

    print(invitation + response)

    print("")

<<<<<<< HEAD
=======
    history.append(invitation + response)
>>>>>>> 57d2455b820e4d5d58e6ca9c21f0986fcc540529

#while True:
go()