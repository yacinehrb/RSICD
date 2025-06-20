import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import random

class ImageCaptionDataset(Dataset):
    def __init__(self, json_path, vocab, transform=None, freq_threshold=5):
        self.vocab = vocab
        self.transform = transform
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Initialize vocabulary if not already built
        if len(self.vocab) == 4:  # Only has <PAD>, <SOS>, <EOS>, <UNK>
            all_captions = []
            for entry in self.data:
                all_captions.extend(entry["captions"])
            self.vocab.build_vocabulary(all_captions)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = random.choice(item['captions'])
        numericalized = [self.vocab.stoi["<SOS>"]] + \
                       self.vocab.numericalize(caption) + \
                       [self.vocab.stoi["<EOS>"]]
        
        return image, torch.tensor(numericalized)
