from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

#model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
model_dir = r"C:\Users\harri\.cache\modelscope\hub\models\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B"
print(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="cuda" # auto
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
prompt = "帮我写一个二分查找法"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2000
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)