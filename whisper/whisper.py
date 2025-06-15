import torch
import os
from transformers import (AutoProcessor,
                          pipeline,
                          AutoModelForSpeechSeq2Seq)
import torch
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

local_model_path = "../../LLM/whisper-large-v3"

if not os.path.exists(local_model_path):
    print(f"Error: Local model path '{local_model_path}' does not exist.")
    exit()

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    local_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(local_model_path)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
print(f"Sample audio file: {sample['path']}\n\n")

result = pipe(sample, return_timestamps=True, generate_kwargs={
    "language": "chinese",
    "task": "translate"})

print(f'\n\nllm response: {result["text"]}')

