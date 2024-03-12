import torch
import open_clip 
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e32')
tokenizer = open_clip.get_tokenizer('ViT-L-14')

image = preprocess(Image.open("/home/anakin/skola/leto/KNN/image.png")).unsqueeze(0)
text = tokenizer(["andrej babis", "cock", "car"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

