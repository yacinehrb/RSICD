import torch
import pickle

def save_checkpoint(model, optimizer, epoch, vocab, filename="checkpoint.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'vocab': vocab
    }, filename)

# Add this to save vocab separately
def save_vocab(vocab, filename="vocab.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({'itos': vocab.itos, 'stoi': vocab.stoi}, f)