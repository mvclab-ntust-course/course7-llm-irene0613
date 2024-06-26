# Week 7  
## Homework 1 - Try difference LoRA configs  
訓練方法主要分成以下幾個步驟，詳細內容請看[Transformers_peft](https://github.com/mvclab-ntust-course/course7-llm-irene0613/blob/main/Transformers_peft.ipynb)
1. 模型和分詞器  
    * 使用`distilbert-base-cased`作為基礎模型  
    * 使用相同的模型檢查點初始化分詞器  
2. 數據準備  
    * 使用`datasets`庫加載 IMDB 電影評論數據集，其數據集包含 50,000 條電影評論，這些評論被標註為正面或負面  
    * 從 IMDB 數據集中創建一個小規模的訓練和驗證數據集(比例為4：1)，並將每條評論截斷為前 50 個tokens   
    * 對剛創建好的小規模訓練和驗證數據集進行分詞處理，分詞過程以批處理方式進行，每次處理 16 個樣本     
3. 訓練參數  
    * 批量大小（batch size）：每次迭代，模型會計算這一批數據的損失，並根據這個損失進行權重更新，因此**批量大小的選擇會影響訓練速度和模型性能**  
    * 學習率（learning rate）：控制模型權重**更新的步伐**，是影響訓練過程中收斂速度和效果的重要參數  
    * 評估策略（evaluation strategy）：確保模型在訓練過程中不過擬合或欠擬合，並提供即時反饋以調整訓練參數，在本次作業中是按輪數評估，每訓練完一個epoch進行一次評估  
    * 訓練輪數（number of epochs）：控制模型在訓練數據集上的訓練次數，確保模型有足夠的機會學習數據中的模式  
4. Trainer  
    * 使用 Hugging Face 的 Trainer 管理訓練過程  
    * 負責模型初始化、訓練參數設置、訓練&驗證數據集設置、分詞器設置與使用自定義評估指標  
5. LoRA & IA3 配置與其相對結果  **LoraConfig_1**  
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
