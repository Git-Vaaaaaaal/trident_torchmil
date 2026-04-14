import openslide
from trident import AnyToTiffConverter
import os
import pandas as pd

# Installer libvips et installer la variable d'environnement
vipsbin = r'C:\\Users\\valbo\\Downloads\\vips-dev-w64-all-8.18.0\\vips-dev-8.18\\bin'

# Ajouter au PATH ET enregistrer le dossier DLL avant d'importer pyvips
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
os.add_dll_directory(vipsbin)  # ← clé pour Python 3.8+

import pyvips

marker_list = ["BCL2", "BCL6", "HE", "MUM1", "MYC", "CD10"]
output = "extract_tma_tiff_img"
os.makedirs(output, exist_ok=True)


for marker in marker_list :
    print("ok ")
    input_dir = os.path.join("extract_tma", marker)
    list_img = os.listdir(input_dir)
    df = pd.DataFrame(columns=["wsi", "mpp"])
    df["wsi"] = list_img
    df["mpp"] = 0.2535
    name = f"{marker}_list.csv"
    df.to_csv(name)
    mark_out = os.path.join(output, marker)
    converter = AnyToTiffConverter(job_dir=mark_out, bigtiff=False)
    converter.process_all(input_dir=input_dir, mpp_csv=name, downscale_by=1)
    print(f"{marker} est terminé")