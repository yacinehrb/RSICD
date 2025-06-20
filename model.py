import torch
import torch.nn as nn
import torchvision.models as models
import random

# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size, train_CNN=False):
#         super(EncoderCNN, self).__init__()
#         self.train_CNN = train_CNN

#         # Load pretrained ResNet50
#         self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

#         # Remove the final fully connected layer
#         modules = list(self.resnet.children())[:-1]  # remove the last fc layer
#         self.resnet = nn.Sequential(*modules)

#         # Add embedding projection
#         self.linear = nn.Linear(2048, embed_size)  # 2048 is the output feature size of ResNet50
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)

#         # Freeze or unfreeze CNN parameters
#         for param in self.resnet.parameters():
#             param.requires_grad = train_CNN

#     def forward(self, images):
#         with torch.set_grad_enabled(self.train_CNN):
#             features = self.resnet(images)  # [B, 2048, 1, 1]
#         features = features.view(features.size(0), -1)  # [B, 2048]
#         features = self.dropout(self.relu(self.linear(features)))  # [B, embed_size]
#         return features

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN

        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Remove classifier, keep the feature extractor
        self.mobilenet = mobilenet.features  # output shape: [B, 1280, 7, 7]

        # Global average pooling to reduce to [B, 1280]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear projection to embed_size
        self.linear = nn.Linear(1280, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Freeze or unfreeze MobileNet
        for param in self.mobilenet.parameters():
            param.requires_grad = train_CNN

    def forward(self, images):
        with torch.set_grad_enabled(self.train_CNN):
            features = self.mobilenet(images)       # [B, 1280, H, W]
            features = self.pool(features)          # [B, 1280, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 1280]
        features = self.dropout(self.relu(self.linear(features)))  # [B, embed_size]
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.teacher_forcing_ratio = 0.5

    def forward(self, features, captions=None, teacher_forcing_ratio=None):
        batch_size = features.size(0)
        
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio
        
        # Initialize hidden state with features
        hiddens, states = self.lstm(features.unsqueeze(1))
        
        if captions is not None:
            # Training mode with teacher forcing
            seq_len = captions.shape[1] - 1  # exclude <SOS>
            outputs = torch.zeros(batch_size, seq_len, self.linear.out_features)
            
            # First input is always <SOS>
            input = self.embed(captions[:, 0]).unsqueeze(1)
            
            for t in range(seq_len):
                hiddens, states = self.lstm(input, states)
                output = self.linear(hiddens.squeeze(1))
                outputs[:, t] = output
                
                # Decide if we use teacher forcing
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing and t < seq_len - 1:
                    input = self.embed(captions[:, t+1]).unsqueeze(1)
                else:
                    input = self.embed(output.argmax(1)).unsqueeze(1)
            
            return outputs
        else:
            # Inference mode
            pass

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions=None, teacher_forcing_ratio=None):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions, teacher_forcing_ratio)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            # features = self.encoderCNN(image.unsqueeze(0))
            features = self.encoderCNN(image)
            inputs = features.unsqueeze(1)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(inputs, states)
                output = self.decoderRNN.linear(hiddens.squeeze(1))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
                    
                inputs = self.decoderRNN.embed(predicted).unsqueeze(1)
            
        return [vocabulary.itos[idx] for idx in result_caption]
