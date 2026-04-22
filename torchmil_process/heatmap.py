
import torch
import h5py
import os
from trident import OpenSlideWSI, visualize_heatmap
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory as patch_encoder_factory

# a. Load WSI to process
job_dir = './tutorial-3/heatmap_viz'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
slide = OpenSlideWSI(slide_path='./CPTAC-CCRCC_v1/CCRCC/C3L-00418-22.svs', lazy_init=False)

# b. Run segmentation Changer segementation
segmentation_model = segmentation_model_factory("hest")
geojson_contours = slide.segment_tissue(segmentation_model=segmentation_model, job_dir=job_dir, device=device)

# c. Run patch coordinate extraction
coords_path = slide.extract_tissue_coords(
    target_mag=20,
    patch_size=512,
    save_coords=job_dir,
    overlap=256, 
)

# d. Run patch feature extraction
patch_encoder = patch_encoder_factory("conch_v15").eval().to(device)
patch_features_path = slide.extract_patch_features(
    patch_encoder=patch_encoder,
    coords_path=coords_path,
    save_features=os.path.join(job_dir, f"features_conch_v15"),
    device=device
)

#  e. Run inference 
with h5py.File(patch_features_path, 'r') as f:
    coords = f['coords'][:]
    patch_features = f['features'][:]
    coords_attrs = dict(f['coords'].attrs)

batch = {'features': torch.from_numpy(patch_features).float().to(device).unsqueeze(0)}
logits, attention = model(batch, return_raw_attention=True)

# f. generate heatmap
heatmap_save_path = visualize_heatmap(
    wsi=slide,
    scores=attention.cpu().numpy().squeeze(),  
    coords=coords,
    vis_level=1,
    patch_size_level0=coords_attrs['patch_size_level0'],
    normalize=True,
    num_top_patches_to_save=10,
    output_dir=job_dir
)