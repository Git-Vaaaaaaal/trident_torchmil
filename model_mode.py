from trident.slide_encoder_models import ABMILSlideEncoder, CHIEFSlideEncoder, FeatherSlideEncoder, GigaPathSlideEncoder, MadeleineSlideEncoder
from trident.slide_encoder_models import ThreadsSlideEncoder, TitanSlideEncoder, PRISMSlideEncoder, MeanSlideEncoder
from torchmil.models import ABMIL, CLAM_SB, DSMIL, TransMIL, DTFDMIL


def options(mode=int):
    if mode == 0 : #Threads
        PATCH_ENCODER = "conch_v15"
        encoder = ThreadsSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 1 : #Titan
        PATCH_ENCODER = "conch_v15"
        encoder = TitanSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 2 : #Prism
        PATCH_ENCODER = "virchow"
        encoder = PRISMSlideEncoder
        PATCH_SIZE = 224
        embedding_level = 2560
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 3 : #Chief
        PATCH_ENCODER = "ctranspath"
        encoder = CHIEFSlideEncoder
        PATCH_SIZE = 256
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 4 : #Gigapath
        PATCH_ENCODER = "virchow"
        encoder = GigaPathSlideEncoder
        PATCH_SIZE = 256
        embedding_level = 1536
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 5 : #Madeleine
        PATCH_ENCODER = "conch_v1"
        encoder = MadeleineSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 512
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 6 : #Prism
        PATCH_ENCODER = "conch_v15"
        encoder = FeatherSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    else :
        return print("Outvalue : between 0 to 6")
    

def options_torchmil(mode=int, marker=str, encoder=str):
    #Dictionnaire de taille de vecteur (embedding)
    dict_encoder = {
        "ThreadsSlideEncoder" : 768,
        "TitanSlideEncoder" : 768,
        "PRISMSlideEncoder" : 2560,
        "CHIEFSlideEncoder" : 768,
        "GigaPathSlideEncoder" : 1536,
        "MadeleineSlideEncoder" : 512,
        "FeatherSlideEncoder" : 768
    }
    if mode == 0 : #ABMIL
        slide_labels_dir = f"extracted\\{encoder}\\slide_features_extraction\\{marker}"
        embedding_level = dict_encoder[encoder]
        model = ABMIL(in_dim=int(embedding_level), att_dim=128)
        return slide_labels_dir, embedding_level, model
    elif mode == 1 : #CLAM
        slide_labels_dir = f"extracted\\{encoder}\\slide_features_extraction\\{marker}"
        embedding_level = dict_encoder[encoder]
        model = CLAM_SB(in_dim=int(embedding_level))
        return slide_labels_dir, embedding_level, model
    elif mode == 2 : #TransMIL
        slide_labels_dir = f"extracted\\{encoder}\\slide_features_extraction\\{marker}"
        embedding_level = dict_encoder[encoder]
        model = TransMIL(in_dim=int(embedding_level))
        return slide_labels_dir, embedding_level, model
    elif mode == 3 : #DSMIL
        slide_labels_dir = f"extracted\\{encoder}\\slide_features_extraction\\{marker}"
        embedding_level = dict_encoder[encoder]
        model = DSMIL(in_dim=int(embedding_level))
        return slide_labels_dir, embedding_level, model
    elif mode == 4 : #DTFDMIL
        slide_labels_dir = f"extracted\\{encoder}\\slide_features_extraction\\{marker}"
        embedding_level = dict_encoder[encoder]
        model = DTFDMIL(in_dim=int(embedding_level))
        return slide_labels_dir, embedding_level, model
    else :
        return print("Outvalue : between 0 to 4")
    

#BINARY CLASSIFICATION DATASET MODIFIE
import numpy as np
import torch
import warnings
import h5py
import os

from torchmil.datasets import ProcessedMILDataset


class BinaryClassificationDataset(ProcessedMILDataset):
    r"""
    Dataset for binary classification MIL problems.
    Extends ProcessedMILDataset with dynamic filtering by bag_name
    from a DataFrame (labels_path), avoiding any file duplication.
    """

    def __init__(
        self,
        features_path: str,
        labels_path,  # Accepte un DataFrame OU un chemin str
        inst_labels_path: str = None,
        coords_path: str = None,
        bag_names: list = None,
        bag_keys: list = ["X", "Y", "y_inst", "adj", "coords"],
        dist_thr: float = 1.5,
        adj_with_dist: bool = False,
        norm_adj: bool = True,
        load_at_init: bool = True,
        verbose: bool = True,
    ) -> None:

        # Si labels_path est un DataFrame, on filtre les bag_names depuis celui-ci
        if hasattr(labels_path, "to_csv"):
            self._labels_df = labels_path.copy()

            # bag_names déduits du DataFrame si non fournis
            if bag_names is None:
                bag_names = self._labels_df["bag_name"].tolist()

            # Filtrage : on garde uniquement les patients dont le .h5 existe
            existing = [
                name for name in bag_names
                if os.path.exists(os.path.join(features_path, f"{name}.h5"))
            ]
            missing = len(bag_names) - len(existing)
            if missing > 0 and verbose:
                print(f"[WARNING] {missing} fichiers .h5 introuvables et ignorés")

            bag_names = existing

            # Sauvegarde temporaire du CSV pour satisfaire ProcessedMILDataset
            self._tmp_labels_path = os.path.join(features_path, "_tmp_labels.csv")
            self._labels_df[self._labels_df["bag_name"].isin(bag_names)].to_csv(
                self._tmp_labels_path, index=False
            )
            labels_path = self._tmp_labels_path

        super().__init__(
            features_path=features_path,
            labels_path=labels_path,
            inst_labels_path=inst_labels_path,
            coords_path=coords_path,
            bag_names=bag_names,
            bag_keys=bag_keys,
            dist_thr=dist_thr,
            adj_with_dist=adj_with_dist,
            norm_adj=norm_adj,
            load_at_init=load_at_init,
            verbose=verbose,
        )

    def _fix_inst_labels(self, inst_labels):
        if inst_labels is not None:
            while inst_labels.ndim > 1:
                inst_labels = np.squeeze(inst_labels, axis=-1)
        return inst_labels

    def _fix_labels(self, labels):
        labels = np.squeeze(labels)
        return labels

    def _load_inst_labels(self, name):
        inst_labels = super()._load_inst_labels(name)
        inst_labels = self._fix_inst_labels(inst_labels)
        return inst_labels

    def _load_labels(self, name):
        labels = super()._load_labels(name)
        labels = self._fix_labels(labels)
        return labels

    def _consistency_check(self, bag_dict, name):
        if "Y" in bag_dict:
            if "y_inst" in bag_dict:
                if bag_dict["Y"] != (bag_dict["y_inst"]).max():
                    if self.verbose:
                        msg = f"Instance labels (max(y_inst)={(bag_dict['y_inst']).max()}) are not consistent with bag label (Y={bag_dict['Y']}) for bag {name}. Setting all instance labels to -1 (unknown)."
                        warnings.warn(msg)
                    bag_dict["y_inst"] = np.full((bag_dict["X"].shape[0],), -1)
            else:
                if bag_dict["Y"] == 0:
                    bag_dict["y_inst"] = np.zeros(bag_dict["X"].shape[0])
                else:
                    if self.verbose:
                        msg = f"Instance labels not found for bag {name}. Setting all to -1."
                        warnings.warn(msg)
                    bag_dict["y_inst"] = np.full((bag_dict["X"].shape[0],), -1)
        return bag_dict

    def _load_bag(self, name: str) -> dict[str, torch.Tensor]:
        bag_dict = super()._load_bag(name)
        bag_dict = self._consistency_check(bag_dict, name)
        return bag_dict