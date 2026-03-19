import cv2
import numpy as np


def apply_threshold_235(input_path: str, output_path: str):
    # Charger l'image TIFF
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError("Impossible de charger l'image")

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Application du threshold à 235
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    # Appliquer le masque sur l'image originale
    result = cv2.bitwise_and(image, image, mask=mask)

    # Sauvegarde en TIFF
    cv2.imwrite(output_path, result)

    return output_path


img = apply_threshold_235("tiff_img\\BCL2\\13906.tiff","13906_thresh.tiff")