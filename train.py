import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from model import CNNtoRNN
from ImageCap import ImageCaptionDataset
from Vocab_builder import Vocabulary
from Colate import MyCollate
from utils import save_checkpoint, save_vocab
import torch.nn as nn
import numpy as np

def train_model():
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
    ])

    # Initialize vocabulary and datasets
    vocab = Vocabulary(freq_threshold=5)
    train_dataset = ImageCaptionDataset("RSICD/train/train_captions.json", vocab, transform)
    val_dataset = ImageCaptionDataset("RSICD/valid/valid_captions.json", vocab, transform)

    # Data loaders
    pad_idx = vocab.stoi["<PAD>"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=MyCollate(pad_idx),
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=MyCollate(pad_idx),
        num_workers=4
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model initialization
    embed_size = 512
    hidden_size = 512  
    vocab_size = len(vocab)
    num_layers = 4
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr= 3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)

    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')

    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, captions in progress_bar:
            images, captions = images.to(device), captions.to(device)
            
            outputs = model(images, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to(device), captions.to(device)
                outputs = model(images, captions)
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        # scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        np.save('T_loss.npy', train_loss_list)
        np.save('V_loss.npy', val_loss_list)

        
        # Save best model
        if val_loss < best_val_loss:
            save_checkpoint(model, optimizer, epoch, vocab, "best_model.pth")
            save_vocab(vocab, "vocab.pkl")  # Save vocabulary
            best_val_loss = val_loss

if __name__ == "__main__":
    train_model()


