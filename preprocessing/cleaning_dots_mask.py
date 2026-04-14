"""
Clean les maks apres application du threshold a la main
"""

from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage


# ──────────────────────────────────────────────
# CONFIGURATION — modifiez ces valeurs
# ──────────────────────────────────────────────

MIN_SIZE_BLACK = 800              # Particule noire  < cette valeur → devient blanche
MIN_SIZE_WHITE = 100                # Zone blanche     < cette valeur → devient noire
CONNECTIVITY   = 2                 # 2 = 8-voisins (diagonales), 1 = 4-voisins strict

marker_list = ["MYC"] #"BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"

# ──────────────────────────────────────────────


def remove_small_black_particles(binary: np.ndarray, min_size: int, connectivity: int) -> np.ndarray:
    """
    Supprime les petites composantes NOIRES (True) isolées.
    Les particules plus petites que min_size pixels sont mises à False (blanc).
    """
    structure = np.ones((3, 3)) if connectivity == 2 else None
    labeled, num_features = ndimage.label(binary, structure=structure)

    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))

    keep = np.zeros(num_features + 1, dtype=bool)
    for i, size in enumerate(sizes, start=1):
        keep[i] = size >= min_size

    return keep[labeled]


def remove_small_white_particles(binary: np.ndarray, min_size: int, connectivity: int) -> np.ndarray:
    """
    Supprime les petites zones BLANCHES (False) isolées.
    Les zones plus petites que min_size pixels sont mises à True (noir).
    """
    # On inverse le masque pour travailler sur les blancs comme s'ils étaient noirs
    inverted = ~binary

    structure = np.ones((3, 3)) if connectivity == 2 else None
    labeled, num_features = ndimage.label(inverted, structure=structure)

    sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))

    keep = np.zeros(num_features + 1, dtype=bool)
    for i, size in enumerate(sizes, start=1):
        keep[i] = size >= min_size

    # Les zones blanches trop petites (non conservées) redeviennent noires
    white_kept = keep[labeled]
    return ~white_kept  # On ré-inverse pour retrouver le sens original (True = noir)


def clean_mask(input_path: Path, output_path: Path,
               min_size_black: int, min_size_white: int, connectivity: int):
    """
    Applique les deux passes de nettoyage sur un fichier TIFF
    et sauvegarde le résultat en préservant la compression originale.
    """
    # Chargement — conservation des métadonnées de compression
    img = Image.open(input_path)
    compression = img.info.get("compression", None)
    tag_info = img.tag_v2 if hasattr(img, "tag_v2") else {}

    arr = np.array(img.convert("L"))
    binary = arr < 128  # True = noir, False = blanc

    # Passe 1 : suppression des petites particules noires
    binary = remove_small_black_particles(binary, min_size_black, connectivity)
    removed_black = int(np.sum(arr < 128) - np.sum(binary))

    # Passe 2 : suppression des petites zones blanches
    binary = remove_small_white_particles(binary, min_size_white, connectivity)
    removed_white = int(np.sum(~(arr < 128)) - np.sum(~binary))

    # Sauvegarde avec la compression originale
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = np.where(binary, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(result, mode="L")

    save_kwargs = {"format": "TIFF"}
    if compression:
        save_kwargs["compression"] = compression
    if tag_info:
        save_kwargs["tiffinfo"] = tag_info

    out_img.save(output_path, **save_kwargs)

    return removed_black, removed_white


def main():
    for marker in marker_list:
        INPUT_FOLDER   = Path(f"output_mask_tiff\\{marker}\\")
        output_folder = INPUT_FOLDER.parent / (INPUT_FOLDER.name + "_cleaned")

        tiff_files = sorted(
            f for f in INPUT_FOLDER.rglob("*")
            if f.suffix.lower() in {".tif", ".tiff"}
        )

        if not tiff_files:
            print(f"Aucun fichier TIFF trouvé dans : {INPUT_FOLDER.resolve()}")
            return

        print(f"Dossier source      : {INPUT_FOLDER.resolve()}")
        print(f"Dossier sortie      : {output_folder.resolve()}")
        print(f"Min. particules noires  : {MIN_SIZE_BLACK} px")
        print(f"Min. zones blanches     : {MIN_SIZE_WHITE} px")
        print(f"Fichiers TIFF       : {len(tiff_files)}\n")
        print(f"  {'Fichier':<40} {'Noires supprimées':>18} {'Blanches supprimées':>20}")
        print(f"  {'─'*40} {'─'*18} {'─'*20}")

        for f in tiff_files:
            out = output_folder / f.relative_to(INPUT_FOLDER)
            removed_black, removed_white = clean_mask(
                f, out, MIN_SIZE_BLACK, MIN_SIZE_WHITE, CONNECTIVITY
            )
            print(f"  {f.name:<40} {removed_black:>16} px {removed_white:>18} px")

        print(f"\nTerminé — fichiers sauvegardés dans : {output_folder.resolve()}")


if __name__ == "__main__":
    main()