import os
import ctypes

# Installer libvips et installer la variable d'environnement
vipsbin = r'C:\\Users\\valbo\Downloads\\vips-dev-w64-all-8.18.0\\vips-dev-8.18\\bin'

# Ajouter au PATH ET enregistrer le dossier DLL avant d'importer pyvips
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
os.add_dll_directory(vipsbin)  # ← clé pour Python 3.8+

import pyvips  # seulement APRÈS les lignes ci-dessus

def png_to_svs(input_png, output_svs):
    # Charger l'image PNG
    image = pyvips.Image.new_from_file(input_png, access="sequential")

    # Sauvegarder en TIFF pyramidal compatible SVS
    image.tiffsave(
        output_svs,
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        compression="jpeg",
        Q=90,
        bigtiff=True
    )

    print(f"Converti : {input_png} → {output_svs}")


# Exemple pour un dossier
input_folder = "pipe_trident_torchmil"
type_tma = ["BCL2", "BCL6", "HE", "MUM1", "MYC"]
output_folder = "output_svs"

os.makedirs(output_folder, exist_ok=True)

for marker in type_tma:
    path = os.path.join(input_folder, marker)
    os.makedirs(path, exist_ok=True)
    for file in os.listdir(path):
        if file.lower().endswith(".png"):
            input_path = os.path.join(path, file)
            output_path = os.path.join(path, file.replace(".png", ".svs"))
            png_to_svs(input_path, output_path)




