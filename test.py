import torch
import gc
import os
import requests
from PIL import Image
from io import BytesIO
from models_registry import MODEL_REGISTRY
from google.colab import userdata


def run_all_tests():
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    print("\n🚀 STARTING MODEL ZOO TEST 🚀\n")
    img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg/560px-Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg"

    try:
        print("Downloading test image...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(img_url,
                                headers=headers,
                                stream=True,
                                timeout=10
                                )
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print("✅ Loaded test Chest X-Ray image.\n")
    except Exception as e:
        print(f"⚠️ Could not download X-Ray ({e}). Using fallback red image.")
        image = Image.new('RGB', (224, 224), color='red')

    for model_name, ModelClass in MODEL_REGISTRY.items():
        print(f"--- 🧪 Testing {model_name} ---")

        model_instance = None
        try:
            model_instance = ModelClass()

            if model_instance.requires_prompt:
                if "Med" in model_name or "MAIRA" in model_name:
                    prompt = "Describe the findings in this chest X-ray."
                else:
                    prompt = "Describe this image."
                output = model_instance.run(image, prompt=prompt)
            else:
                output = model_instance.run(image)

            print(f"✅ OUTPUT: {output}\n")

        except Exception as e:
            print(f"❌ FAILED: {e}\n")

        if model_instance:
            del model_instance
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_all_tests()
