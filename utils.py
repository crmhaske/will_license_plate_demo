import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # https://huggingface.co/microsoft/trocr-base-printed
from PIL import Image

def extract_license_plate_number(image_path):
    with Image.open(image_path) as img:
        image = img.convert('RGB')

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text


    
   