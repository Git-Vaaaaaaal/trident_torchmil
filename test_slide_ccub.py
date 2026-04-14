import os
import torch
import segmentation_models_pytorch
from trident import Processor
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry
import numpy as np 
import cv2
import tiff

# =========================
# CONFIGURATION (MODIFIABLE)
# =========================

SLIDE_ENCODER = "feather"

GPU = 0
#TASK = "seg"  # "seg", "coords", "feat", "all"


SKIP_ERRORS = False
MAX_WORKERS = None
BATCH_SIZE = 64

WSI_CACHE = None
CACHE_BATCH_SIZE = 32

WSI_EXT = None
CUSTOM_MPP_KEYS = None
CUSTOM_LIST_OF_WSIS = None
READER_TYPE = None
SEARCH_NESTED = False

# Segmentation
SEGMENTER = "hest"
SEG_CONF_THRESH = 0.35
REMOVE_HOLES = False
REMOVE_ARTIFACTS = False
REMOVE_PENMARKS = False
SEG_BATCH_SIZE = None

# Patching
MAG = 20.0
PATCH_SIZE = 16
OVERLAP = 0
MIN_TISSUE_PROPORTION = 0.0
COORDS_DIR = None

# Feature extraction
PATCH_ENCODER = None
PATCH_ENCODER_CKPT_PATH = None

FEAT_BATCH_SIZE = None


# =========================
# INITIALIZATION
# =========================

def initialize_processor():
    return Processor(
        job_dir=JOB_DIR,
        wsi_source=WSI_DIR,
        wsi_ext=WSI_EXT,
        wsi_cache=WSI_CACHE,
        skip_errors=SKIP_ERRORS,
        custom_mpp_keys=CUSTOM_MPP_KEYS,
        custom_list_of_wsis=CUSTOM_LIST_OF_WSIS,
        max_workers=MAX_WORKERS,
        reader_type=READER_TYPE,
        search_nested=SEARCH_NESTED,
    )


# =========================
# TASK EXECUTION
# =========================


# Threshold
def threshold_tiff(path, thresh=230):
    img = tiff.imread(path)

    kernel1 = np.ones((3,3), np.uint8)
    kernel2 = np.ones((3,3), np.uint8)

    
    
    # Si multi-canaux → grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gestion TIFF 16 bits → normalisation en 8 bits
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    img = cv2.blur(img,(9,9))
    
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel2)

    return binary.astype(np.uint8)

def run_task(processor, task):
    if task == 'seg':
        from trident.segmentation_models.load import segmentation_model_factory

        seg_device = "cpu" if SEGMENTER == "otsu" else f"cuda:{GPU}"

        segmentation_model = segmentation_model_factory(
            SEGMENTER,
            confidence_thresh=SEG_CONF_THRESH,
        )

        if REMOVE_ARTIFACTS or REMOVE_PENMARKS:
            artifact_remover_model = segmentation_model_factory(
                'hest',
                remove_penmarks_only=REMOVE_PENMARKS and not REMOVE_ARTIFACTS
            )
        else:
            artifact_remover_model = None

        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue=not REMOVE_HOLES,
            artifact_remover_model=artifact_remover_model,
            batch_size=SEG_BATCH_SIZE if SEG_BATCH_SIZE else BATCH_SIZE,
            device=seg_device,
        )

    elif task == 'coords':
        processor.run_patching_job(
            target_magnification=MAG,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            saveto=COORDS_DIR,
            min_tissue_proportion=MIN_TISSUE_PROPORTION
        )

    elif task == 'feat':
        if SLIDE_ENCODER is None:
            from trident.patch_encoder_models.load import encoder_factory

            encoder = encoder_factory(PATCH_ENCODER, weights_path=PATCH_ENCODER_CKPT_PATH)

            processor.run_patch_feature_extraction_job(
                coords_dir=COORDS_DIR or f'{MAG}x_{PATCH_SIZE}px_{OVERLAP}px_overlap',
                patch_encoder=encoder,
                device=f'cuda:{GPU}',
                saveas='h5',
                batch_limit=FEAT_BATCH_SIZE if FEAT_BATCH_SIZE else BATCH_SIZE,
            )
        else:
            from trident.slide_encoder_models.load import encoder_factory

            encoder = encoder_factory(SLIDE_ENCODER)

            processor.run_slide_feature_extraction_job(
                slide_encoder=encoder,
                coords_dir=COORDS_DIR or f'{MAG}x_{PATCH_SIZE}px_{OVERLAP}px_overlap',
                device=f'cuda:{GPU}',
                saveas='h5',
                batch_limit=FEAT_BATCH_SIZE if FEAT_BATCH_SIZE else BATCH_SIZE,
            )

    else:
        raise ValueError(f"Invalid task: {task}")


# =========================
# MAIN
# =========================

from huggingface_hub import login
from env import hf_token

login(hf_token)

device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

list_marker = ["BCL2", "BCL6", "HE", "MUM1", "MYC", "CD10"]
slide_encoder_list = ["feather", "madeleine", "gigapath", "chief", "prism", "titan", "threads"]

path_contour = "job_dir"
path_wsi = "wsi_source"
path = "trident_torchmil/results"
for marker in list_marker:
    marker_path = os.path.join(path, marker)
    JOB_DIR = os.path.join(marker_path, path_contour) #output, les geojson doivent etre places dans results
    WSI_DIR = os.path.join(marker_path, path_wsi)
    processor = initialize_processor()
    tasks = ['coords', 'feat'] #if TASK == 'all' else [TASK]

    for task in tasks:
        print(f"Running task: {task}")
        run_task(processor, task)


