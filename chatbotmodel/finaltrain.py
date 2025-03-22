import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
import random

# Optimized Hyperparameters
BATCH_SIZE = 64
HIDDEN_SIZE = 256
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 500
DROPOUT_PROB = 0.5

def main():
    # Load and process data
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    def augment_text(words):
        if len(words) > 1 and random.random() < 0.3:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return words

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            words = tokenize(pattern)
            all_words.extend(words)
            xy.append((words, tag))
            augmented_words = augment_text(words.copy())
            xy.append((augmented_words, tag))

    ignore_words = ['?', ',', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Create training data
    X = []
    y = []
    for pattern_sentence, tag in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X.append(bag)
        y.append(tags.index(tag))

    X = np.array(X)
    y = np.array(y, dtype=np.int64)

    # Split data first
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Class balancing using training set only
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(y_train),
        replacement=True
    )

    class ChatDataset(Dataset):
        def __init__(self, X_data, y_data):
            self.x_data = torch.from_numpy(X_data).float()
            self.y_data = torch.from_numpy(y_data).long()
            self.n_samples = len(X_data)

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    train_dataset = ChatDataset(X_train, y_train)
    val_dataset = ChatDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(
        input_size=X.shape[1],
        hidden_size=HIDDEN_SIZE,
        num_classes=len(tags),
        context_size=0,
        dropout_prob=DROPOUT_PROB
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.3
    )

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "input_size": X.shape[1],
                "output_size": len(tags),
                "hidden_size": HIDDEN_SIZE,
                "all_words": all_words,
                "tags": tags
            }, "best_model.pth")
        
        if epoch - best_epoch > 50:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:04d}/{NUM_EPOCHS} | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f} | '
                  f'Train Acc: {train_acc:.4f} | '
                  f'Val Acc: {val_acc:.4f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print(f'\nBest Validation Accuracy: {best_acc:.4f} at epoch {best_epoch+1}')
    print('Training completed. Best model saved to best_model.pth')

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()