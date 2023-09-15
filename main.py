import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageCaptioning, GPT2Tokenizer, GPT2LMHeadModel

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16")
model = ViTForImageCaptioning.from_pretrained("google/vit-base-patch16")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
caption_model = GPT2LMHeadModel.from_pretrained("gpt2")

image_path = "path_to_your_image.jpg"
image = Image.open(image_path)

inputs = feature_extractor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

image_features = outputs.last_hidden_state

caption_ids = caption_model.generate(image_features)
caption = tokenizer.decode(caption_ids[0], skip_special_tokens=True)

print("Generated Caption:", caption)