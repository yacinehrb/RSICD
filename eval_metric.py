import json
import torch
from model import CNNtoRNN
import pickle
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import evaluate
import nltk

# nltk.download('wordnet')

# nltk.download('punkt')



bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
# cider = evaluate.load("cider")
bertscore = evaluate.load("bertscore")

# Load vocabulary
def load_vocab(filename="vocab.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    vocab = type('Vocabulary', (), {})()
    vocab.itos = data['itos']
    vocab.stoi = data['stoi']
    return vocab

# Load vocab
vocab = load_vocab(r"results googlenet/vocab.pkl")

# Load model
embed_size = 256
hidden_size = 256
vocab_size = len(vocab.itos)
num_layers = 1

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
checkpoint = torch.load(r"results googlenet/best_model.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def generate_caption(image_path, model, vocab, device="cpu", max_length=50):
    image = load_image(image_path).to(device)
    model.to(device)
    with torch.no_grad():
        caption = model.caption_image(image, vocab, max_length=max_length)
    return ' '.join(caption)

# Load test JSON
with open("RSICD/test/test_captions.json", "r") as f:
    data = json.load(f)

# Evaluation setup
predictions = []
references = []

# Generate captions and prepare for evaluation
for item in tqdm(data):
    image_path = item["image_path"]
    ref_captions = item["captions"]
    pred_caption = generate_caption(image_path, model, vocab)

    predictions.append(pred_caption)
    references.append(ref_captions)  # list of references

# Load metrics


# Compute metrics
print("Calculating BLEU...")
bleu_result = bleu.compute(predictions=predictions, references=references)
print("BLEU:", bleu_result)

print("Calculating METEOR...")
meteor_result = meteor.compute(predictions=predictions, references=references)
print("METEOR:", meteor_result)

print("Calculating ROUGE...")
rouge_result = rouge.compute(predictions=predictions, references=references)
print("ROUGE:", rouge_result)

# print("Calculating CIDEr...")
# cider_result = cider.compute(predictions=predictions, references=references)
# print("CIDEr:", cider_result)

print("Calculating BERTScore (may take time)...")
bertscore_result = bertscore.compute(predictions=predictions, references=[refs[0] for refs in references], lang="en")
print("BERTScore:", {
    "precision": sum(bertscore_result["precision"]) / len(bertscore_result["precision"]),
    "recall": sum(bertscore_result["recall"]) / len(bertscore_result["recall"]),
    "f1": sum(bertscore_result["f1"]) / len(bertscore_result["f1"]),
})
