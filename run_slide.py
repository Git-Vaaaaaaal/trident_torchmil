import os
import torch

from trident import Processor
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry

# =========================
# CONFIGURATION (MODIFIABLE)
# =========================

GPU = 0
TASK = "all"  # "seg", "coords", "feat", "all"

JOB_DIR = "output_slides"
WSI_DIR = "tiff_img/BCL2"
os.makedirs(JOB_DIR, exist_ok=True)

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
SEG_CONF_THRESH = 0.5
REMOVE_HOLES = False
REMOVE_ARTIFACTS = False
REMOVE_PENMARKS = False
SEG_BATCH_SIZE = None

# Patching
MAG = 20.0
PATCH_SIZE = 512
OVERLAP = 0
MIN_TISSUE_PROPORTION = 0.0
COORDS_DIR = None

# Feature extraction
PATCH_ENCODER = None
PATCH_ENCODER_CKPT_PATH = None
SLIDE_ENCODER = "titan"
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
                'grandqc_artifact',
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

def main():
    device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    processor = initialize_processor()

    tasks = ['seg', 'coords', 'feat'] if TASK == 'all' else [TASK]

    for task in tasks:
        print(f"Running task: {task}")
        run_task(processor, task)


if __name__ == "__main__":
    main()
