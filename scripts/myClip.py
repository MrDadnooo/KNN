import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("/home/anakin/skola/leto/KNN/res/cache/processing.2023-09-18/zips/crops/uuid:deea8f32-4d51-4987-8626-4962e8c21957__obrázek_1.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a dog on a boat playing with someone",
                      "nigga on a boat playing with bow and observing distance",
                      "woman on a boat playing with bow"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image, )
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

