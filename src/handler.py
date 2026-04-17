import runpod, torch, os
import transformers.integrations.peft
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from huggingface_hub import login

# --- WORKAROUND: Bypass the PEFT MoE conversion bug ---
if not hasattr(transformers.integrations.peft, "_MOE_TARGET_MODULE_MAPPING"):
    transformers.integrations.peft._MOE_TARGET_MODULE_MAPPING = {}
transformers.integrations.peft._MOE_TARGET_MODULE_MAPPING['llava'] = {}
# ------------------------------------------------------

login(token=os.environ.get("HF_KEY"))

def generate_prompt(text, image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": text},
        ]
    }]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    return messages

# Load model once when worker starts
base_model_id = "google/medgemma-4b-it"
lora_adapter_path = "Hrushikesh-0000/medgemma-4b-it-sft-lora-MRI6k"

processor = AutoProcessor.from_pretrained(base_model_id)

# FIX 1: Load base model first, then apply LoRA adapter on top
base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    dtype=torch.bfloat16,  # FIX 2: use `dtype` instead of deprecated `torch_dtype`
    device_map="cuda",
)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval()

pipe = pipeline(
    "image-text-to-text",
    model=model,           # pass the already-loaded PEFT model
    tokenizer=processor.tokenizer,
    image_processor=processor.image_processor,
    # no device= needed; model is already on CUDA via device_map above
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