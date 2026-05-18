from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id="Qwen/Qwen3-8B",
    cache_dir="/root/autodl-tmp/modelscope"
)

print(model_dir)