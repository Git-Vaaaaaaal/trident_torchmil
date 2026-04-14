#Code temporaire pour retransformer les data manquantes des masks 
#Pour transformer les png en tiff et les mettre dans le dossier output_mask_tiff
import os
from PIL import Image
import pandas as pd



path = "extract_tma"

df_dataset = pd.read_csv(os.path.join(path, "new_dataset_list.csv"))

marker_list = ["BCL2", "BCL6", "HE", "MUM1", "MYC", "CD10"]

sum_photo = 0

for marker in marker_list :
    marker_path = os.path.join(path, marker)
    marker_list = len(os.listdir(marker_path))
    sum_photo += marker_list
    print(f"Le nombre d'images pour le marqueur {marker} est de {marker_list}")
    df_marker = df_dataset[df_dataset["stain"] == marker]
    print(f"Le nombre d'images pour le marqueur {marker} dans le dataset est de {len(df_marker)}")

print(f"Le nombre total d'images est de {sum_photo}")
print(f"Le nombre total d'images dans le dataset est de {len(df_dataset)}")