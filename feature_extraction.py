import os
from pathlib import Path 
import torch 
from PIL import Image
import geopandas as gpd
from IPython.display import display
from huggingface_hub import snapshot_download
import h5py
from trident.slide_encoder_models import ABMILSlideEncoder, CHIEFSlideEncoder, FeatherSlideEncoder, GigaPathSlideEncoder, MadeleineSlideEncoder
from trident.slide_encoder_models import ThreadsSlideEncoder, TitanSlideEncoder, PRISMSlideEncoder, MeanSlideEncoder
from trident.patch_encoder_models import encoder_factory
from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory

#variables
list_markers = ["BCL2"] # "BCL2", "BCL6", "HE", "MUM1", "MYC"
DEVICE = f"cuda:0" if torch.cuda.is_available() else "cpu"
input_dir = "pipe_trident_torchmil"
output_dir = os.makedirs("extracted", exist_ok=True)
verification = False

# b. Create OpenSlideWSI
for marker in list_markers :
    marker_path = os.path.join(input_dir, marker)
    TARGET_MAG = 20
    PATCH_SIZE = 512
    for patient in os.listdir(marker_path):
        patient_id = patient.replace(".svs", "")
        patient_svs = os.path.join(marker_path, patient)
        slide = OpenSlideWSI(slide_path=patient_svs, lazy_init=False) # a partir d'ici == definition

        # c. Run segmentation 
        segmentation_model = segmentation_model_factory("hest")
        geojson_contours = slide.segment_tissue(segmentation_model=segmentation_model, target_mag=10, job_dir=None, device=DEVICE)

        if verification == True :
            # d. Visualize contours
            contour_image = Image.open(os.path.join(f"{marker}_{patient_id}" + ".png", 'contours', patient_svs.replace('.svs', '.jpg')))
            display(contour_image)

            # e. Check contours saved into GeoJSON with GeoPandas
            gdf = gpd.read_file(geojson_contours)
            gdf.head(n=10)

        # a. Run patch coordinate extraction
        coords_output = os.path.join(output_dir, "coordinates")
        os.makedirs(coords_output, exist_ok=True)
        coords_output = os.path.join(coords_output, marker)
        os.makedirs(coords_output, exist_ok=True)
        coords_path = slide.extract_tissue_coords(
            target_mag=TARGET_MAG,
            patch_size=PATCH_SIZE,
            save_coords=os.path.join(coords_output, f"{patient_id}.h5"),
        )

        #Patch encoder fetaures
        PATCH_ENCODER = "uni_v1"
        encoder = encoder_factory(PATCH_ENCODER)
        encoder.eval()
        encoder.to(DEVICE)

        # b. Run UNI feature extraction
        patch_features_dir = os.path.join(output_dir, "patch_features_extraction")
        patch_features_dir = os.path.join(patch_features_dir, marker)

        patch_feats_path = slide.extract_patch_features(
        patch_encoder=encoder, 
        coords_path=coords_path, #Path to the file containing patch coordinates.
        save_features=os.path.join(patch_features_dir, f"{patient_id}.h5"), #path to extracted features
        device=DEVICE
        )

        #Encoder features
        encoder = TitanSlideEncoder
        encoder.eval()
        encoder.to(DEVICE)

        # b. Run UNI feature extraction
        slide_features_dir = os.path.join(output_dir, "slide_features_extraction")
        slide_features_dir = os.path.join(slide_features_dir, marker)

        slide_feats_path = slide.extract_slide_features(
        patch_features_path=os.path.join(patch_features_dir, f"{patient_id}.h5"), #Attend un fichier H5, 
        slide_encoder = encoder,
        save_features=os.path.join(slide_features_dir, f"{patient_id}.h5"), # changer
        device=DEVICE
        )

print("feature extraction works")







