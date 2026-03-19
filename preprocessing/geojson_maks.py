import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import os

def mask_to_geojson(mask_array, image_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]

    binary = np.where(mask_array > 0, 255, 0).astype(np.uint8)

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )

    features = []
    tissue_id = 0

    if contours is not None and len(contours) > 0:
        hierarchy_flat = hierarchy[0]

        for i, contour in enumerate(contours):
            if hierarchy_flat[i][3] != -1:
                continue

            if cv2.contourArea(contour) < MIN_AREA:
                continue

            approx = cv2.approxPolyDP(contour, EPSILON, closed=True) if EPSILON > 0 else contour
            coords = [[float(p[0][0]), float(p[0][1])] for p in approx]
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            rings = [coords]

            child_idx = hierarchy_flat[i][2]
            while child_idx != -1:
                hole = contours[child_idx]
                if cv2.contourArea(hole) >= MIN_AREA:
                    hole_approx = cv2.approxPolyDP(hole, EPSILON, closed=True) if EPSILON > 0 else hole
                    hole_coords = [[float(p[0][0]), float(p[0][1])] for p in hole_approx]
                    if hole_coords[0] != hole_coords[-1]:
                        hole_coords.append(hole_coords[0])
                    rings.append(hole_coords)
                child_idx = hierarchy_flat[child_idx][0]

            features.append({
                "type": "Feature",
                "properties": {"tissue_id": tissue_id},
                "geometry": {"type": "Polygon", "coordinates": rings},
            })
            tissue_id += 1

    geojson = {
        "type": "FeatureCollection",
        "name": image_name,
        "crs": {"type": "name", "properties": {"name": CRS}},
        "features": features,
    }

    out_path = output_dir / f"{image_name}.geojson"
    out_path.write_text(json.dumps(geojson, separators=(",", ":")), encoding="utf-8")
    print(f"  {tissue_id} region(s)  ->  {out_path.name}")
    return out_path


INPUT_DIR  = (f"F:\\imvia\\trident_torchmil\\output")
OUTPUT_DIR = (f"F:\\imvia\\trident_torchmil\\output_geojson")

list_markers = ["BCL2", "BCL6", "HE", "MUM1", "MYC"]

MIN_AREA = 500    # surface minimale en px² pour ignorer le bruit
EPSILON  = 2.0   # simplification du contour (0 = désactivé)
CRS      = "urn:ogc:def:crs:EPSG::3857"

EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}



for marker in list_markers:
    marker_path = os.path.join(INPUT_DIR, marker)
    os.makedirs(marker_path, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, marker)
    os.makedirs(out_path, exist_ok=True)
    img = os.listdir(marker_path)
    for im in img :
        im = os.path.join(marker_path, im)
        arr = np.array(Image.open(im))
        im_path = Path(im)
        mask_to_geojson(arr, im_path.stem, out_path) 
        print(f"{im} features enregistrés.")