# ================================
# 1. Import Necessary Libraries
# ================================
import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from textwrap import wrap
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.model_selection import train_test_split

# For evaluation metrics (BLEU and METEOR)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# ================================
# 2. Set Random Seeds and Device
# ================================
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ================================
# 3. Data Preparation and Vocabulary Building
# ================================
def tokenize(text):
    text = text.lower()
    return re.findall(r'\w+', text)


def load_and_prepare_captions(captions_file):
    """
    Reads a comma-delimited text file with header:
        image_name,comment_number,comment
    Groups captions by image, builds vocabulary, and converts each caption into a sequence.
    """
    caption_df = pd.read_csv(captions_file)
    print("Head of the caption DataFrame:\n", caption_df.head())

    image_captions = defaultdict(list)
    for _, row in caption_df.iterrows():
        image_captions[row['image_name']].append(row['comment'])

    # Build vocabulary
    all_captions = [caption for caps in image_captions.values() for caption in caps]
    all_words = [word for caption in all_captions for word in tokenize(caption)]
    word_counts = Counter(all_words)

    # Special tokens
    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
    word2idx = {token: idx for idx, token in enumerate(special_tokens)}
    idx2word = {idx: token for idx, token in enumerate(special_tokens)}

    vocab_size = 10000  # Adjust as needed
    for idx, (word, _) in enumerate(word_counts.most_common(vocab_size - len(special_tokens)),
                                    start=len(special_tokens)):
        word2idx[word] = idx
        idx2word[idx] = word

    captions_seqs = {}
    max_length = 0
    for img_name, captions in image_captions.items():
        seqs = []
        for caption in captions:
            tokens = ['<start>'] + tokenize(caption) + ['<end>']
            seq = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
            seqs.append(seq)
            max_length = max(max_length, len(seq))
        captions_seqs[img_name] = seqs

    return caption_df, word2idx, idx2word, image_captions, captions_seqs, max_length


# ================================
# 4. Image Transformations and Dataset Setup
# ================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])


def read_image_for_display(path, img_size=224):
    disp_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert('RGB')
    img = disp_transform(img)
    return img.permute(1, 2, 0).numpy()


class FlickrDataset(Dataset):
    def __init__(self, image_dir, image_ids, captions_seqs, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.captions = []
        for img_id in image_ids:
            for caption_seq in captions_seqs[img_id]:
                self.images.append(img_id)
                self.captions.append(caption_seq)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        caption_seq = self.captions[idx]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption_seq = torch.tensor(caption_seq)
        return image, caption_seq


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    targets = torch.zeros(len(captions), max_len).long()
    for i, cap in enumerate(captions):
        targets[i, :len(cap)] = cap
    return images, targets, lengths


# ================================
# 5. Model Definitions: Encoder with Spatial Features & Attention Decoder
# ================================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Load pre-trained ResNet-50 and remove avgpool and fc layers
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]  # keep conv layers up to layer4
        self.resnet_conv = nn.Sequential(*modules)
        # Freeze earlier layers and fine-tune only layers 3 and 4 (indices 6 and 7)
        for name, param in self.resnet_conv.named_parameters():
            # Allow gradients for modules with names containing 'layer3' or 'layer4'
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # Project conv features (2048) to embed_size for each spatial location.
        self.embed = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # Get conv feature maps: (batch, 2048, 7, 7)
        features = self.resnet_conv(images)
        batch_size, channels, feat_h, feat_w = features.size()
        # Reshape to (batch, num_regions, channels)
        features = features.view(batch_size, channels, -1).permute(0, 2, 1)  # (B, 49, 2048)
        # Project each region to embed_size.
        features = self.embed(features)  # (B, 49, embed_size)
        # Apply batch normalization (flatten spatial dim)
        features = features.view(-1, features.size(-1))
        features = self.bn(features)
        features = features.view(batch_size, -1, features.size(-1))
        return features  # (B, 49, embed_size)


class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attn_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(feature_dim + hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, features, hidden):
        hidden_exp = hidden.unsqueeze(1).expand(features.size(0), features.size(1), hidden.size(1))
        concat = torch.cat((features, hidden_exp), dim=2)  # (B, num_regions, feature_dim+hidden_dim)
        energy = torch.tanh(self.attn(concat))  # (B, num_regions, attn_dim)
        scores = self.v(energy).squeeze(2)  # (B, num_regions)
        alpha = torch.softmax(scores, dim=1)  # (B, num_regions)
        context = torch.sum(features * alpha.unsqueeze(2), dim=1)  # (B, feature_dim)
        return context, alpha


class DecoderRNN_Attn(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, dropout=0.5):
        super(DecoderRNN_Attn, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(feature_dim=embed_size, hidden_dim=hidden_size, attn_dim=attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, features, captions):
        embeddings = self.embed(captions)  # (B, max_len, embed_size)
        batch_size = features.size(0)
        num_steps = embeddings.size(1)
        h = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(features.device)
        c = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(features.device)
        outputs = []
        for t in range(num_steps):
            embed_t = embeddings[:, t, :]  # (B, embed_size)
            context, _ = self.attention(features, h)  # (B, embed_size)
            lstm_input = torch.cat((embed_t, context), dim=1)  # (B, 2*embed_size)
            h, c = self.lstm_cell(lstm_input, (h, c))
            h = self.dropout(h)
            output = self.fc(h)  # (B, vocab_size)
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (B, max_len, vocab_size)
        return outputs

    def sample(self, features, word2idx, max_length=20):
        batch_size = features.size(0)
        h = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(features.device)
        c = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(features.device)
        inputs = torch.tensor([word2idx['<start>']] * batch_size).to(features.device)
        sampled_ids = []
        for _ in range(max_length):
            embed = self.embed(inputs)  # (B, embed_size)
            context, _ = self.attention(features, h)  # (B, embed_size)
            lstm_input = torch.cat((embed, context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            outputs = self.fc(h)  # (B, vocab_size)
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.unsqueeze(1))
            inputs = predicted
            if (predicted == word2idx['<end>']).all():
                break
        sampled_ids = torch.cat(sampled_ids, dim=1)  # (B, seq_length)
        return sampled_ids


# ================================
# 6b. Helper Functions for Evaluation and Caption Generation
# ================================
def generate_caption(encoder, decoder, image, word2idx, idx2word, max_length=20):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)  # (1, C, H, W)
        features = encoder(image)  # (1, 49, embed_size)
        sampled_ids = decoder.sample(features, word2idx, max_length)
        sampled_ids = sampled_ids[0].cpu().numpy().tolist()
    caption = [idx2word.get(idx, '<unk>') for idx in sampled_ids]
    caption = [word for word in caption if word not in ['<start>', '<end>', '<pad>']]
    return ' '.join(caption)


def prepare_image2captions(image_ids, captions_seqs, idx2word):
    image2captions = {}
    for img_id in image_ids:
        seqs = captions_seqs[img_id]
        captions_list = []
        for seq in seqs:
            caption = [idx2word.get(idx, '<unk>') for idx in seq]
            caption = [word for word in caption if word not in ['<start>', '<end>', '<pad>']]
            captions_list.append(caption)
        image2captions[img_id] = captions_list
    return image2captions


def evaluate_model(encoder, decoder, data_loader, criterion):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_batches = 0
    with torch.no_grad():
        for images, captions, lengths in data_loader:
            images = images.to(device)
            captions = captions.to(device)
            lengths = torch.tensor(lengths)
            adjusted_lengths = lengths - 1  # exclude <start>
            features = encoder(images)  # (B, 49, embed_size)
            outputs = decoder(features, captions)
            targets = nn.utils.rnn.pack_padded_sequence(captions[:, 1:], adjusted_lengths, batch_first=True,
                                                        enforce_sorted=False)[0]
            outputs = \
                nn.utils.rnn.pack_padded_sequence(outputs, adjusted_lengths, batch_first=True, enforce_sorted=False)[0]
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches


def calculate_bleu(encoder, decoder, image_dir, image_ids, image2captions, transform, word2idx, idx2word):
    encoder.eval()
    decoder.eval()
    references = []
    hypotheses = []
    smoothie = SmoothingFunction().method4
    with torch.no_grad():
        for img_id in image_ids:
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            caption_generated = generate_caption(encoder, decoder, image_tensor, word2idx, idx2word, max_length=20)
            hypotheses.append(caption_generated.split())
            refs = [[word.lower() for word in ref] for ref in image2captions[img_id]]
            references.append(refs)
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    return bleu


def calculate_meteor(encoder, decoder, image_dir, image_ids, image2captions, transform, word2idx, idx2word):
    encoder.eval()
    decoder.eval()
    meteor_scores = []
    with torch.no_grad():
        for img_id in image_ids:
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            caption_generated = generate_caption(encoder, decoder, image_tensor, word2idx, idx2word, max_length=20)
            score = meteor_score(image2captions[img_id], caption_generated.split())
            meteor_scores.append(score)
    return sum(meteor_scores) / len(meteor_scores)


# ================================
# 7. Main Training and Evaluation Pipeline
# ================================
if __name__ == '__main__':
    # Provided paths
    image_dir = r'.\archive\flickr30k_images'
    captions_file = r'.\archive\captions.txt'

    # Load data and build vocabulary
    (caption_df, word2idx, idx2word, image_captions,
     captions_seqs, max_caption_length) = load_and_prepare_captions(captions_file)

    image_names = list(image_captions.keys())
    train_images, temp_images = train_test_split(image_names, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    train_dataset = FlickrDataset(image_dir, train_images, captions_seqs, transform)
    val_dataset = FlickrDataset(image_dir, val_images, captions_seqs, transform)
    test_dataset = FlickrDataset(image_dir, test_images, captions_seqs, transform)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, collate_fn=collate_fn)

    # Hyperparameters
    num_epochs = 10  # Increased epochs for better convergence
    embed_size = 256
    hidden_size = 512
    attention_dim = 256
    vocab_size = len(word2idx)

    # Initialize models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN_Attn(embed_size, hidden_size, vocab_size, attention_dim, dropout=0.5).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    optimizer = optim.Adam([
        {'params': decoder.parameters()},
        {'params': encoder.embed.parameters()},
        {'params': encoder.bn.parameters()},
        # Access layer3 and layer4 via indices: [6] and [7] in the Sequential
        {'params': encoder.resnet_conv[6].parameters(), 'lr': 1e-5},
        {'params': encoder.resnet_conv[7].parameters(), 'lr': 1e-5}
    ], lr=1e-4, weight_decay=1e-4)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    val_image2captions = prepare_image2captions(val_images, captions_seqs, idx2word)
    test_image2captions = prepare_image2captions(test_images, captions_seqs, idx2word)

    train_losses, val_losses, val_bleu_scores, val_meteor_scores = [], [], [], []

    print("\nStarting Training...\n")
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0
        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            lengths = torch.tensor(lengths)
            features = encoder(images)  # (B, 49, embed_size)
            outputs = decoder(features, captions)  # (B, max_len, vocab_size)
            adjusted_lengths = lengths - 1
            targets = nn.utils.rnn.pack_padded_sequence(captions[:, 1:], adjusted_lengths, batch_first=True,
                                                        enforce_sorted=False)[0]
            outputs = \
                nn.utils.rnn.pack_padded_sequence(outputs, adjusted_lengths, batch_first=True, enforce_sorted=False)[0]
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
            optimizer.step()
            epoch_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

        val_loss = evaluate_model(encoder, decoder, val_loader, criterion)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}")

        bleu = calculate_bleu(encoder, decoder, image_dir, val_images, val_image2captions, transform, word2idx,
                              idx2word)
        val_bleu_scores.append(bleu)
        meteor = calculate_meteor(encoder, decoder, image_dir, val_images, val_image2captions, transform, word2idx,
                                  idx2word)
        val_meteor_scores.append(meteor)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation BLEU: {bleu:.4f}, METEOR: {meteor:.4f}\n")

        scheduler.step()

    # Plot training curves
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(epochs_range, val_bleu_scores, label='Validation BLEU Score')
    plt.plot(epochs_range, val_meteor_scores, label='Validation METEOR Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation BLEU and METEOR Scores')
    plt.legend()
    plt.show()

    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')
    print("Models saved to disk.\n")


    # Testing: Display sample test images with generated captions.
    def display_test_images_with_captions(image_ids, num_images=5):
        plt.figure(figsize=(20, 20))
        for i in range(num_images):
            plt.subplot(3, 2, i + 1)
            plt.subplots_adjust(hspace=0.7, wspace=0.3)
            img_id = image_ids[i]
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert('RGB')
            caption = generate_caption(encoder, decoder, transform(image), word2idx, idx2word, max_length=20)
            plt.imshow(image)
            plt.title("\n".join(wrap("Generated Caption: " + caption, 40)))
            plt.axis("off")
        plt.show()


    display_test_images_with_captions(test_images, num_images=5)
