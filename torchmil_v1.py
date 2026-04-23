import os 
import h5py
import openslide
import cv2
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from class_tridentwsi import TridentTMADataset



TRIDENT_DIR = "BCL2" #Path to the directory where trident will save the extracted features.
TIFF_DIR = r"extract_tma_tiff_img\BCL2"   #Path to the directory where the tiff files are stored.

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


patch_size = 16
coords_dir = TRIDENT_DIR + r"\job_dir\20.0x_16px_0px_overlap\patches"
patch_labels_dir = TRIDENT_DIR + r"\job_dir\20.0x_16px_0px_overlap\features_conch_v15"

wsi_name = "13901"

wsi_path = os.path.join(TIFF_DIR, wsi_name + ".tiff")
slide = openslide.OpenSlide(wsi_path)


coords_path = os.path.join(coords_dir, wsi_name + "_patches.h5")
with h5py.File(coords_path, "r") as f:
    print(list(f.keys()))
inst_coords = h5py.File(coords_path, "r")["coords"][:]

patch_features_path = os.path.join(patch_labels_dir, wsi_name + ".h5")
print(patch_features_path)

with h5py.File(patch_features_path, "r") as f:
    print(list(f.keys()))
    features = f["features"][:]   # ✅ au lieu de patch_labels

print(f"Number of patches: {len(inst_coords)}")
print(f"First 5 patch coordinates: {inst_coords[:5]}")

slide_label_dict = {
    "13901": 1,   # ex: positif
    # "13902": 0, etc.
}

slide_label = slide_label_dict[wsi_name]


if len(inst_coords) != len(features):
    raise ValueError("Mismatch between number of patches and features")

fig, axs = plt.subplots(1, 5, figsize=(20, 4))
for i in range(5):
    x, y = inst_coords[i]
    patch_pil = slide.read_region((x, y), 0, (patch_size, patch_size))
    patch = np.array(patch_pil)[:, :, :3]  # Convert to RGB
    patch = cv2.resize(patch, (patch_size, patch_size))
    axs[i].imshow(patch)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title(
    f"Patch {i+1}\n(x={x}, y={y})\nSlide Label={slide_label}",
    fontsize=16
    )
plt.tight_layout()
plt.show()


#New cells 
def read_wsi_patches(slide, inst_coords, patch_size=512, resize_size=50):
    bag_len = len(inst_coords)
    patches_list = []
    row_list = []
    column_list = []
    for i in tqdm.tqdm(range(bag_len), desc="Reading_patches"):
        x, y = inst_coords[i]
        patch_pil = slide.read_region((x, y), 0, (patch_size, patch_size))
        patch = np.array(patch_pil)[:, :, :3]  # Convert to RGB
        patch_resized = cv2.resize(patch, (resize_size, resize_size))
        patches_list.append(patch_resized)
        row = int(y/ patch_size)
        column = int(x/ patch_size)
        row_list.append(row)
        column_list.append(column)

    row_array = np.array(row_list)
    column_array = np.array(column_list)

    row_array = row_array - row_array.min()
    column_array = column_array - column_array.min()

    return patches_list, row_array, column_array


patches_list, row_array, column_array = read_wsi_patches(slide, inst_coords, patch_size=512, resize_size=50)




#New cells
from torchmil.datasets import TridentWSIDataset
from sklearn.model_selection import train_test_split
import pandas as pd


patch_labels_path = TRIDENT_DIR + "patch_labels/"
feature_extractor = "conch_v15"


bcl2_labels_path = "bcl2_torchmil_test.csv"
df_data = pd.read_csv(bcl2_labels_path)

train_df = df_data[df_data["stain"] == "BCL2"]
test_df = df_data[df_data["stain"] == "BCL2"]


train_wsis = train_df["new_patient_id"].astype(str).tolist()

test_wsis = test_df["new_patient_id"].astype(str).tolist()

#Path to the features
base_path = os.path.join(TRIDENT_DIR, "job_dir", "20.0x_16px_0px_overlap")


dataset = TridentTMADataset(
    base_path=base_path + r"\\", 
    labels_path=bcl2_labels_path,
    feature_extractor=feature_extractor,
    patch_labels_path=patch_labels_path,
    wsi_names=train_wsis,
    bag_keys=["X", "Y", "y_inst", "adj", "coords"],
    patch_size=patch_size,
    load_at_init=True,
    wsi_name_col="new_patient_id",
    wsi_label_col="status",
    )

dataset_test = TridentTMADataset(
    base_path=base_path + r"\\",
    labels_path=bcl2_labels_path,
    feature_extractor=feature_extractor,
    patch_labels_path=patch_labels_path,
    wsi_names=test_wsis,
    bag_keys=["X", "Y", "y_inst", "adj", "coords"],
    patch_size=patch_size,
    load_at_init=True,
    wsi_name_col="wsi_name",
    wsi_label_col="wsi_label",
    )


bag_labels = dataset.get_bag_labels()
idx = list(range(len(bag_labels)))
val_prop = 0.2
idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=42, stratify=bag_labels)
train_dataset = dataset.subset(idx_train)
val_dataset = dataset.subset(idx_val)
test_dataset = dataset_test.subset(list(range(len(dataset_test))))

#Print one bag
bag = train_dataset[0]
print(f"Bag type:",type(bag))
for key in bag.keys():
    print(key, bag[key].shape)


#New cells
from torchmil.data import collate_fn

batch_size = 1

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


it = iter(train_loader)
batch = next(it)
print(f"Batch type:", type(batch))
for key in batch.keys():
    print(key, batch[key].shape)


#New cells
from torchmil.nn import masked_softmax


class ABMIL(torch.nn.Module):
    def __init__(self, emb_dim, att_dim):
        super().__init__()

        # Feature extractor
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, emb_dim),
        )

        self.fc1 = torch.nn.Linear(emb_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1)

        self.classifier = torch.nn.Linear(emb_dim, 1)

    def forward(self, X, mask, return_att=False):
        X = self.mlp(X)  # (batch_size, bag_size, emb_dim)
        H = torch.tanh(self.fc1(X))  # (batch_size, bag_size, att_dim)
        att = torch.sigmoid(self.fc2(H))  # (batch_size, bag_size, 1)
        att_s = masked_softmax(att, mask)  # (batch_size, bag_size, 1)
        # att_s = torch.nn.functional.softmax(att, dim=1)
        X = torch.bmm(att_s.transpose(1, 2), X).squeeze(1)  # (batch_size, emb_dim)
        Y_pred = self.classifier(X).squeeze(1)  # (batch_size,)
        if return_att:
            return Y_pred, att_s
        else:
            return Y_pred


model = ABMIL(emb_dim=256, att_dim=128)
print(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ABMIL(emb_dim=256, att_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

model = model.to(device)
criterion = criterion.to(device)  

def train(dataloader, epoch):
    model.train()

    sum_loss = 0.0
    sum_correct = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch["X"], batch["mask"])
        loss = criterion(out, batch["Y"].float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        sum_loss += loss.item()
        pred = (out > 0).float()
        sum_correct += (pred == batch["Y"]).sum().item()
        sum_loss += loss.item()

    print(
        f"[Epoch {epoch}] Train, train/loss: {sum_loss / len(dataloader)}, 'train/bag/acc': {sum_correct / len(dataloader.dataset)}"
    )


def val(dataloader, epoch):
    model.eval()

    sum_loss = 0.0
    sum_correct = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch["X"], batch["mask"])
        loss = criterion(out, batch["Y"].float())

        sum_loss += loss.item()
        pred = (out > 0).float()
        sum_correct += (pred == batch["Y"]).sum().item()
        sum_loss += loss.item()

    print(
        f"[Epoch {epoch}] Validation, val/loss: {sum_loss / len(dataloader)}, 'val/bag/acc': {sum_correct / len(dataloader.dataset)}"
    )


model = model.to(device)
for epoch in range(20):
    train(train_loader, epoch + 1)
    val(test_loader, epoch + 1)


#New cells 
from torchmil.visualize import patches_to_canvas, draw_heatmap_wsi

canvas = patches_to_canvas(patches_list, row_array, column_array, 50)

canvas_with_patch_labels = draw_heatmap_wsi(canvas, features, 50, row_array, column_array)

fig, axs = plt.subplots(1, 2, figsize=(6, 6))
axs[0].imshow(canvas)
axs[0].set_title("TMA", fontsize=16)
axs[1].imshow(canvas_with_patch_labels)
axs[1].set_title("TMA with Patch Labels", fontsize=16)
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
