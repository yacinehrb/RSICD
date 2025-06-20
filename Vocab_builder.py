import re

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = len(self.itos)

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

                if frequencies[token] == self.freq_threshold:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]
    def load_vocab(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            vocab_dict = pickle.load(f)
        self.itos = vocab_dict['itos']
        self.stoi = vocab_dict['stoi']
        
    def save_vocab(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'itos': self.itos, 'stoi': self.stoi}, f)