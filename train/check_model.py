from transformers import AutoTokenizer

model_name = "/root/autodl-tmp/modelscope/Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=True
)

messages = [
    {"role": "system", "content": "你是一名专业、礼貌、遵守平台规则的电商客服助手。"},
    {"role": "user", "content": "用户问：这个食品拆封了还能退吗？"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("tokenizer 加载成功！")
print(text[:500])