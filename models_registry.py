# models_registry.py
import torch
from PIL import Image
from transformers import (
    pipeline,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
import os


# ==========================================
# 1. Base Class
# ==========================================
class BaseModel:
    description = "Base model"
    requires_prompt = False

    def run(self, image: Image.Image, prompt: str = None):
        raise NotImplementedError("Implement run() in subclass")


# ==========================================
# 2. BLIP Base
# ==========================================
class BLIPBaseModel(BaseModel):
    description = "Great for fast inference on general images"
    requires_prompt = False

    def __init__(self):
        self.pipe = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=0
                )

    def run(self, image: Image.Image, prompt: str = None):
        return self.pipe(image)[0]['generated_text']


# ==========================================
# 3. BLIP Large
# ==========================================
class BLIPLargeModel(BaseModel):
    description = "Get more detailed than BLIP Base"
    requires_prompt = False

    def __init__(self):
        self.pipe = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-large",
                device=0
                )

    def run(self, image: Image.Image, prompt: str = None):
        return self.pipe(image)[0]['generated_text']


# ==========================================
# 4. Qwen2-VL (7B Instruct)
# ==========================================
class QwenVLModel(BaseModel):
    description = "Great for getting detailed captions on general images"
    requires_prompt = True

    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        print(f"Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

    def run(self, image: Image.Image, prompt: str = "Describe this image"):
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}
        ]
        text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
                )[0]


# ==========================================
# 5. MedGemma 4B
# ==========================================
class MedGemmaModel(BaseModel):
    description = "Analyse medical images."
    requires_prompt = True

    def __init__(self, model_id="google/medgemma-4b-it"):
        print(f"Loading {model_id}...")
        self.pipe = pipeline(
            "image-text-to-text", model=model_id, device=0,
            torch_dtype=torch.bfloat16, token=os.environ.get('HF_TOKEN')
        )

    def run(self,
            image: Image.Image,
            prompt: str = "Describe this medical image."
            ):
        messages = [
                {"role": "system", "content":
                 [{"type": "text", "text": prompt}]
                 },
                {"role": "user", "content":
                 [{"type": "image", "image": image}]
                 }
            ]
        output = self.pipe(text=messages, max_new_tokens=200)
        if isinstance(output, list) and len(output) > 0:
            last_msg = output[0].get("generated_text", [{}])[-1]
            if isinstance(last_msg, dict):
                return last_msg.get('content', str(output))
        return str(output)


# ==========================================
# EXPORT REGISTRY
# ==========================================
MODEL_REGISTRY = {
    "BLIP Base": BLIPBaseModel,
    "BLIP Large": BLIPLargeModel,
    "Qwen2-VL": QwenVLModel,
    "MedGemma-4B": MedGemmaModel,
}
