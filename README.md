# Week 7  
## Homework - Try difference LoRA configs  
**LoraConfig_1**  
```
peft_config_1 = LoraConfig(
    lora_alpha=16,          
    lora_dropout=0.1,         
    r=64,              
    bias="none",          
    task_type="SEQ_CLS",      
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin"], 
)
```
<img src="https://github.com/mvclab-ntust-course/course7-llm-irene0613/blob/main/image/lora_1.png" width="500px"><br>  
  
**LoraConfig_2**
```
# 改變目標模組
peft_config_2 = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin", "ffn.lin1", "ffn.lin2"],  # 新增"ffn.lin1", "ffn.lin2" 前饋神經網絡
)
```
<img src="https://github.com/mvclab-ntust-course/course7-llm-irene0613/blob/main/image/lora_2.png" width="500px"><br>  
  
**LoraConfig_3**
```
# 改變lora_alpha、lora_dropout、r
peft_config_3 = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.2,
    r=32,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
)
```
<img src="https://github.com/mvclab-ntust-course/course7-llm-irene0613/blob/main/image/lora_3.png" width="500px"><br>  
  
**IA3Config**
```
peft_config_4 = IA3Config(
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin", "ffn.lin1", "ffn.lin2"],
    feedforward_modules=["ffn.lin1", "ffn.lin2"],
)
```
<img src="https://github.com/mvclab-ntust-course/course7-llm-irene0613/blob/main/image/IA3.png" width="500px"><br>  
