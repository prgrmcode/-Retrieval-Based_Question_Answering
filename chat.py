import torch
from transformers import pipeline

# source: https://huggingface.co/meta-llama/Llama-3.2-1B

model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

output = pipe("Once upon a time there lived Ali Baba")
print(output)