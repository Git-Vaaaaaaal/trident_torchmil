import os
import h5py
import openslide
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchmil.datasets import TridentWSIDataset, BinaryClassificationDataset
from torchmil.data import collate_fn
from torchmil.utils import Trainer
from torchmil.models import ABMIL
import torchmetrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from csv_extractor import split_csv


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

TRIDENT_DIR = "extracted" # Path to the directory where TRIDENT will save the extracted patches and their corresponding feature vectors.
TIFF_DIR = "output_svs\\BCL2" # Path to the directory where the original WSIs in TIFF format are stored.

def get_metrics(device):
    return {
        "acc": torchmetrics.Accuracy(task="binary").to(device),
        "auc": torchmetrics.AUROC(task="binary").to(device),
    }

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

patch_size = 512
coords_dir = TRIDENT_DIR + "patches/"
patch_labels_dir = TRIDENT_DIR + "patch_labels/"

""" wsi_name = "test_016"
# wsi_name = "tumor_005"

wsi_path = os.path.join(TIFF_DIR, wsi_name + ".tif")
slide = openslide.OpenSlide(wsi_path)

coords_path = os.path.join(coords_dir, wsi_name + "_patches.h5")
inst_coords = h5py.File(coords_path, "r")["coords"][:]ss

patch_labels_path = os.path.join(patch_labels_dir, wsi_name + ".h5")
patch_labels = h5py.File(patch_labels_path, "r")["patch_labels"][:]
 """
patch_labels_path = TRIDENT_DIR + "patch_labels/"
feature_extractor = "conch_v15"

#Import de df dataset_list
csv_path = "dataset_list.csv"
dataset_csv = pd.read_csv(csv_path)

print("Etape 01")

#Seulement pour train sur BCL2 et titan
marker_list = ["BCL2"]
image_folder = "output_svs\\BCL2"
train_csv, test_csv = split_csv(dataset_csv, "", marker_list, image_folder)

list_rmv = ["tma_id", "stain", "xs", "ys", "xe", "ye"]
dict_cl_name = {"patient_id" : "bag_name", "status" : "label"}

train_csv = train_csv.drop(columns=list_rmv)
test_csv = test_csv.drop(columns=list_rmv)

train_csv = train_csv.drop(columns=list_rmv)
test_csv = test_csv.drop(columns=list_rmv)
#Fin spe BCL2 et Titan


train_labels_path = "BCL2_train.csv"
train_wsis = pd.read_csv(train_labels_path)["wsi_name"].tolist()

test_labels_path = "BCL2_test.csv"
test_wsis = pd.read_csv(test_labels_path)["wsi_name"].tolist()

print("Etape 02")

#Classification binaire
dataset_train = BinaryClassificationDataset(
    features_path="features/", #Mettre les features dans un dossier
    labels_path=train_csv #Changer en un df avec seulement deux colonnes : bag_name et label
)

dataset_test = BinaryClassificationDataset(
    features_path="features/",
    labels_path=test_csv #Changer en un df avec seulement deux colonnes : bag_name et label
)

# Split the dataset into train and validation sets
bag_labels = dataset_train.get_bag_labels()
idx = list(range(len(bag_labels)))
val_prop = 0.2
idx_train, idx_val = train_test_split(
    idx, test_size=val_prop, random_state=1234, stratify=bag_labels
)
train_dataset = dataset_train.subset(idx_train)
val_dataset = dataset_train.subset(idx_val)

# Print one bag
bag = train_dataset[0]
print("Bag type:", type(bag))
for key in bag.keys():
    print(key, bag[key].shape)


batch_size = 1

# Create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_dataloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

print("Etape 03")

it = iter(train_dataloader)
batch = next(it)
print("Batch type:", type(batch))
for key in batch.keys():
    print(key, batch[key].shape)


model = ABMIL(in_dim=768, att_dim=128)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    metrics_dict=get_metrics(device),
    obj_metric="acc",
    device=device,
    disable_pbar=False,
    verbose=False,
)

trainer.train(
    max_epochs=40, train_dataloader=train_dataloader, val_dataloader=val_dataloader
)


inst_pred_list = []
y_inst_list = []
Y_pred_list = []
Y_list = []

model = model.to(device)
model.eval()

probs_list =[]

for batch in test_dataloader:
    batch = batch.to(device)

    X = batch["X"].to(device)
    mask = batch["mask"].to(device)
    adj = batch["adj"].to(device)
    Y = batch["Y"]

    # predict bag label using our model
    out = model(X, mask)
    probs = torch.sigmoid(out)   # probabilité
    Y_pred = (probs > 0.5).float()
    probs_list.append(probs)
    Y_pred_list.append(Y_pred)
    Y_list.append(Y)

Y_pred = torch.cat(Y_pred_list).cpu().numpy()
Y = torch.cat(Y_list).cpu().numpy()
probs = torch.cat(probs_list).cpu().numpy()

print("test/bag/auc:", roc_auc_score(Y, probs))

print(f"test/bag/acc: {accuracy_score(Y_pred, Y)}")
print(f"test/bag/f1: {f1_score(Y_pred, Y)}")
plot_confusion_matrix(Y, Y_pred)

