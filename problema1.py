import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path.cwd()

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


    IMAGE_PATH = BASE_DIR / "imagenes" / "monedas.jpg"


if __name__ == "__main__":
    main()


try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

IMAGE_PATH = BASE_DIR / "imagenes" / "monedas.jpg"


#img_binary = _, binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img = cv2.imread(str(IMAGE_PATH))

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
g3 = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT)
imshow(g3, title="Top-Hat - OpenCV")



k = 5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
print(kernel)

fd = cv2.dilate(g3, kernel)
fe = cv2.erode(g3, kernel)
fmg = cv2.morphologyEx(g3, cv2.MORPH_GRADIENT, kernel)
np.unique(fd)

plt.figure()
ax1 = plt.subplot(221); plt.imshow(g3, cmap='gray'), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(fd, cmap='gray'), plt.title('Dilatacion'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(fe, cmap='gray'), plt.title('Erosion'), plt.xticks([]), plt.yticks([])
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(fmg, cmap='gray'), plt.title('Gradiente Morfologico'), plt.xticks([]), plt.yticks([])
plt.show(block=False)

img_gray = cv2.cvtColor(fmg, cv2.COLOR_BGR2GRAY)
umbral, g1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure(figsize=(6,6))
plt.imshow(g1, cmap='gray')
plt.title("Contornos detectados (kernel 3x3)")
plt.axis("off")
plt.show()

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(g1, connectivity=8)
plt.figure(), plt.imshow(labels, cmap='gray'), plt.show(block=False)


H, W = g1.shape[:2]
RHO_TH = 0.8    # Factor de forma (rho)
AREA_TH = 500   # Umbral de area
aux = np.zeros_like(labels)
labeled_image = cv2.merge([aux, aux, aux])

# --- Clasificación ---------------------------------------------------------------------
# Clasifico en base al factor de forma
for i in range(1, num_labels):

    # --- Remuevo las celulas que tocan el borde de la imagen -----------------
    if (stats[i, cv2.CC_STAT_LEFT] == 0 or 
        stats[i, cv2.CC_STAT_TOP] == 0 or 
        stats[i, cv2.CC_STAT_HEIGHT] + stats[i, cv2.CC_STAT_TOP] == H or 
        stats[i, cv2.CC_STAT_WIDTH] + stats[i, cv2.CC_STAT_LEFT] == W):
        continue

    # --- Remuevo celulas con area chica --------------------------------------
    if (stats[i, cv2.CC_STAT_AREA] < AREA_TH):
        continue

    # --- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Calculo Rho ---------------------------------------------------------
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)
    rho = 4 * np.pi * area/(perimeter**2)
    flag_circular = rho > RHO_TH

    # --- Calculo cantidad de huecos ------------------------------------------
    all_contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = len(all_contours) - 1

    # --- Muestro por pantalla el resultado -----------------------------------
    print(f"Objeto {i:2d} --> Circular: {flag_circular}  /  Huecos: {holes}")

contours_3x3, hierarchy_3x3 = cv2.findContours(g1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
min_area = 800

contours_filtrados = [c for c in contours_3x3 if cv2.contourArea(c) >= min_area]



img_contours_3x3 = fd.copy()
cv2.drawContours(img_contours_3x3, contours_filtrados, -1, (0, 255, 0), 2)



plt.figure(figsize=(6,6))
plt.imshow(img_contours_3x3, cmap='gray')
plt.title("Contornos detectados (kernel 3x3)")
plt.axis("off")
plt.show()

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