## Set Environment

```
sh set_up_env.sh
```

## Train

```
python example_train_<peft_method>_5k_fp16.py
```

## Evaluate

```
python evaluation_batch.py <peft_method>
```


|             | lora | qlora | prompt tuning | ptuning |
| ----------- | ---- | ----- | ------------- | - |
| centralized | :white_check_mark:     |  :white_check_mark:     |  :white_check_mark:         | :white_check_mark:  |    
|             |      |       |              |   |
