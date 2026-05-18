| Experiment | Target | Quant | DType | r | Eval Loss | Eval Acc | Time(s) | Max Alloc GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qlora4bit_all_linear_r16_bf16_alpha32 | all_linear | 4bit | bf16 | 16 | 2.7323 | 0.5319 | 1987.6185 | 10.5027 |
| qlora4bit_qk_r16_bf16_alpha32 | qk | 4bit | bf16 | 16 | 3.4319 | 0.4051 | 1453.5629 | 9.9666 |
| lora_qk_r4_fp16_alpha8 | qk | none | fp16 | 4 | 3.5653 | 0.3656 | 991.1620 | 17.5748 |
| lora_qk_r16_fp16_alpha32 | qk | none | fp16 | 16 | 3.4056 | 0.4069 | 954.9254 | 17.5748 |
| lora_qk_r64_fp16_alpha128 | qk | none | fp16 | 64 | 3.2029 | 0.4630 | 975.5707 | 17.5748 |
