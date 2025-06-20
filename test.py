import torch
from model import CNNtoRNN
import pickle
from torchvision import transforms
from PIL import Image

# Load vocabulary
def load_vocab(filename="vocab.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    vocab = type('Vocabulary', (), {})()  # create an empty object
    vocab.itos = data['itos']
    vocab.stoi = data['stoi']
    return vocab

vocab = load_vocab(r"results resnet 2\vocab.pkl")

# Initialize model with same parameters used during training
embed_size = 512
hidden_size = 512
vocab_size = len(vocab.itos)
num_layers = 4

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)

# Load checkpoint
checkpoint = torch.load(r"results resnet 2\best_model.pth", map_location=torch.device('cpu'), weights_only= False)  # use CPU or "cuda" if available
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image

def generate_caption(image_path, model, vocab, device="cpu", max_length=50):
    image = load_image(image_path)  # add batch dimension
    image = image.to(device)
    model.to(device)

    caption = model.caption_image(image, vocab, max_length=max_length)
    caption_str = ' '.join(caption)
    return caption_str

# Example usage:
test_image_path = r"RSICD\test\test_images\00623.jpg"
caption = generate_caption(test_image_path, model, vocab, device="cpu")
print("Generated Caption:", caption)
