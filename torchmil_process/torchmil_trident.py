import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from trident.slide_encoder_models import ABMILSlideEncoder

# Set deterministic behavior
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=768, n_heads=1, head_dim=512, dropout=0., gated=True, hidden_dim=256):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=input_feature_dim, 
            n_heads=n_heads, 
            head_dim=head_dim, 
            dropout=dropout, 
            gated=gated
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features).squeeze(1)
        
        if return_raw_attention:
            return logits, attn
        
        return logits

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryClassificationModel().to(device)

# Custom dataset
class H5Dataset(Dataset):
    def __init__(self, feats_path, df, split, num_features=512):
        self.df = df[df["fold_0"] == split]
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with h5py.File(os.path.join(self.feats_path, row['slide_id'] + '.h5'), "r") as f:
            features = torch.from_numpy(f["features"][:])

        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED))  # Oversampling
            features = features[indices]

        label = torch.tensor(row["BAP1_mutation"], dtype=torch.float32)
        return features, label

# Create dataloaders
feats_path = './tutorial-3/cptac_ccrcc/20x_512px_0px_overlap/features_conch_v15'
batch_size = 8
train_loader = DataLoader(H5Dataset(feats_path, df, "train"), batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
test_loader = DataLoader(H5Dataset(feats_path, df, "test"), batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))

# Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=4e-4)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.
    for features, labels in train_loader:
        features, labels = {'features': features.to(device)}, labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")




# Evaluation
model.eval()
all_labels, all_outputs = [], []
correct = 0
total = 0

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = {'features': features.to(device)}, labels.to(device)
        outputs = model(features)
        
        # Convert logits to probabilities and binary predictions
        predicted = (outputs > 0).float()  # Since BCEWithLogitsLoss expects raw logits
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_outputs.append(outputs.cpu().numpy())  
        all_labels.append(labels.cpu().numpy())

# Compute AUC
all_outputs = np.concatenate(all_outputs)
all_labels = np.concatenate(all_labels)
auc = roc_auc_score(all_labels, all_outputs)

# Compute accuracy
accuracy = correct / total
print(f"Test AUC: {auc:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

