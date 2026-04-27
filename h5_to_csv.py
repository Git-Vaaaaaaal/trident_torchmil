import h5py
import pandas as pd
from pathlib import Path
import os

""" def h5_to_csv(h5_path: str) -> None:
    h5_path = Path(h5_path)
    output_dir = h5_path.parent / h5_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    def export_dataset(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        csv_path = output_dir / Path(name.lstrip("/")).with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(obj[()])
        df.to_csv(csv_path, index=False)
        print(f"✓ {name} → {csv_path}")

    with h5py.File(h5_path, "r") as f:
        f.visititems(export_dataset)
 """
slide_features = r"BCL2\job_dir\20.0x_16px_0px_overlap\slide_features_feather"
patches_coords = r"BCL2\job_dir\20.0x_16px_0px_overlap\patches"
features_conch = r"BCL2\job_dir\20.0x_16px_0px_overlap\features_conch_v15"

""" 
for path in os.listdir(features_conch):
    if path.endswith(".h5"):
        h5_to_csv(os.path.join(features_conch, path))
 """


import pandas as pd
from pathlib import Path

# For slide h5 files
def merge_patient_csvs(folder: str) -> None:
    folder = Path(folder)
    records = []

    for patient_dir in sorted(folder.iterdir()):
        csv_file = patient_dir / "features.csv"
        if not csv_file.exists():
            continue
        patient_id = patient_dir.name
        features = pd.read_csv(csv_file, header=None).squeeze().tolist()
        records.append([patient_id] + features)

    n_features = max(len(r) - 1 for r in records)
    columns = ["patient"] + [f"feature_{i+1}" for i in range(n_features)]

    df = pd.DataFrame(records, columns=columns)
    output_path = folder / "patients_features.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ {len(records)} patients → {output_path}")


merge_patient_csvs(slide_features)




#For patch h5 files
def merge_patient_csvs(folder: str, h5_dir: str) -> None:
    folder = Path(folder)
    patient_id = folder.name

    coords  = pd.read_csv(folder / "coords.csv",   header=None, names=["x", "y"])
    features = pd.read_csv(folder / "features.csv", header=None,
                           names=[f"feature_{i}" for i in range(768)])

    df = pd.concat([coords, features], axis=1)

    output_path = Path(h5_dir) / f"{patient_id}.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ {patient_id} → {output_path}  ({len(df)} tuiles)")



for path in os.listdir(features_conch):
        merge_patient_csvs((os.path.join(features_conch, path)), features_conch)


