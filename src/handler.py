import runpod, torch, requests, os
from PIL import Image
from transformers import pipeline, AutoProcessor
from huggingface_hub import login

login(token=os.environ.get("HF_KEY"))

def generate_prompt(text, image):
    messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ]}]
    if image!=None:
        messages[0]["content"].append({"type": "image", "image": image})
    return messages

# Load model once when worker starts

base_model_id = "google/medgemma-4b-it"
lora_adapter_path = "Hrushikesh-0000/medgemma-4b-it-sft-lora-MRI6k"

processor = AutoProcessor.from_pretrained(base_model_id)

pipe = pipeline(
    "image-text-to-text",
    model=lora_adapter_path,
    processor=processor,
    device="cuda",
    # Note: We omit device="cuda" here because device_map="cuda" handled it during model loading
    torch_dtype=torch.bfloat16,
)

pipe.model.generation_config.do_sample = False
pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
processor.tokenizer.padding_side = "left"

def handler(job):
    # Extract input from the job
    job_input = job["input"]
    text = job_input.get("text")

    # Validate input
    if not text:
        return {"error": "No text provided for analysis."}

    # Run inference
    result = pipe(
        generate_prompt(text, None),
        max_new_tokens=20,
        batch_size=64,
        return_full_text=True,
    )

    # Return formatted results
    return result[0]['generated_text'][-1]

runpod.serverless.start({"handler": handler})