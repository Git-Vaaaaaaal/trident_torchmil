import os
import h5py

def explore_h5_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".h5"):
                file_path = os.path.join(root, file)
                print(f"\n📁 Fichier : {file_path}")
                
                try:
                    with h5py.File(file_path, "r") as f:
                        def print_structure(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                print(f"  📄 Dataset : {name} | shape={obj.shape}")
                            elif isinstance(obj, h5py.Group):
                                print(f"  📂 Groupe  : {name}")

                        f.visititems(print_structure)

                except Exception as e:
                    print(f"  ❌ Erreur lecture : {e}")

# 👉 Mets ici ton dossier racine contenant les .h5
root_path = r"BCL2/job_dir/20.0x_16px_0px_overlap/slide_features_feather"

explore_h5_files(root_path)
