import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path.cwd()

    IMAGE_PATH = BASE_DIR / "imagenes" / "monedas.jpg"


if __name__ == "__main__":
    main()


try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

IMAGE_PATH = BASE_DIR / "imagenes" / "monedas.jpg"

img = cv2.imread(str(IMAGE_PATH))

if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

val, img_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img_medianBlur = cv2.medianBlur(img_thresh, 5)

# Aplicar desenfoque gaussiano con diferentes tamaños de kernel
blurred_3x3 = cv2.GaussianBlur(img_medianBlur, (3, 3), 2)
blurred_5x5 = cv2.GaussianBlur(img_medianBlur, (5, 5), 2)
blurred_9x9 = cv2.GaussianBlur(img_medianBlur, (9, 9), 2)
blurred_13x13 = cv2.GaussianBlur(img_medianBlur, (13, 13), 2)

low = int(0.5 * val)

# Obtener bordes binarios con Canny (ajusta los umbrales si es necesario)
edges_3x3 = cv2.Canny(blurred_3x3, low, val, apertureSize=5, L2gradient=True)
edges_5x5 = cv2.Canny(blurred_5x5, low, val, apertureSize=5, L2gradient=True)
edges_9x9 = cv2.Canny(blurred_9x9, low, val, apertureSize=5, L2gradient=True)
edges_13x13 = cv2.Canny(blurred_13x13, low, val, apertureSize=5, L2gradient=True)

contours_3x3, hierarchy_3x3 = cv2.findContours(edges_3x3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_5x5, hierarchy_5x5 = cv2.findContours(edges_5x5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_9x9, hierarchy_9x9 = cv2.findContours(edges_9x9, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_13x13, hierarchy_13x13 = cv2.findContours(edges_13x13, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# dibujar contornos sobre copias de la imagen original (BGR)
img_bgr = img.copy()
img_contours_3x3 = img_bgr.copy()
img_contours_5x5 = img_bgr.copy()
img_contours_9x9 = img_bgr.copy()
img_contours_13x13 = img_bgr.copy()

cv2.drawContours(img_contours_3x3, contours_3x3, -1, (0, 255, 0), 2)
cv2.drawContours(img_contours_5x5, contours_5x5, -1, (0, 255, 0), 2)
cv2.drawContours(img_contours_9x9, contours_9x9, -1, (0, 255, 0), 2)
cv2.drawContours(img_contours_13x13, contours_13x13, -1, (0, 255, 0), 2)

# convertir a RGB para matplotlib
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_contours_3x3_rgb = cv2.cvtColor(img_contours_3x3, cv2.COLOR_BGR2RGB)
img_contours_5x5_rgb = cv2.cvtColor(img_contours_5x5, cv2.COLOR_BGR2RGB)
img_contours_9x9_rgb = cv2.cvtColor(img_contours_9x9, cv2.COLOR_BGR2RGB)
img_contours_13x13_rgb = cv2.cvtColor(img_contours_13x13, cv2.COLOR_BGR2RGB)

items = [
    ("Original", img_rgb, None),
    ("Contornos 3x3", img_contours_3x3_rgb, len(contours_3x3)),
    ("Contornos 5x5", img_contours_5x5_rgb, len(contours_5x5)),
    ("Contornos 9x9", img_contours_9x9_rgb, len(contours_9x9)),
    ("Contornos 13x13", img_contours_13x13_rgb, len(contours_13x13)),
]

# rellenar hasta 9 con imágenes en blanco para completar la cuadrícula
h, w, _ = img_rgb.shape
blank = np.ones_like(img_rgb) * 255
while len(items) < 9:
    items.append(("Vacío", blank, None))

fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
axes = axes.ravel()

for ax, (title, im, cnt) in zip(axes, items):
    ax.imshow(im)
    ax.set_title(title if cnt is None else f"{title} ({cnt})")
    ax.axis("off")

plt.show(block=False)