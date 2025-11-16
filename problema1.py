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

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
g3 = cv2.morphologyEx(img, kernel=se, op=cv2.MORPH_TOPHAT)


k = 5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))

fmg = cv2.morphologyEx(g3, cv2.MORPH_GRADIENT, kernel)


img_gray = cv2.cvtColor(fmg, cv2.COLOR_BGR2GRAY)
umbral, g1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
g1 = cv2.GaussianBlur(g1, (5, 5), 2)


# --- Parámetros ---
RHO_TH = 0.8        # umbral de circularidad
MIN_AREA = 3000     # umbral de área mínimo

# --- Obtengo contornos externos ---
contours_3x3, hierarchy_3x3 = cv2.findContours(g1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# --- Filtro contornos por área ---
contours_filtrados = [c for c in contours_3x3 if cv2.contourArea(c) >= MIN_AREA]

# Imagen del mismo tamaño que g1, toda en negro
mask_contornos = np.zeros_like(g1)

# Dibujar los contornos en la máscara
cv2.drawContours(mask_contornos, contours_filtrados, -1, 255, -1)

mask_cerrada = cv2.threshold(mask_contornos, 127, 255, cv2.THRESH_BINARY)[1]
edges = cv2.Canny(mask_cerrada, 50, 150)

circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=300,
    param1=50,
    param2=40,
    minRadius=10,
    maxRadius=200
)

img_hough = edges.copy()   # imagen para dibujar

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0,:]:
        cv2.circle(img_hough, (x, y), r, (255,255,255), -1)   # círculo relleno

# --- Mostrar ---
plt.figure(figsize=(6,6))
if len(img_hough.shape) == 2:
    plt.imshow(img_hough, cmap='gray')
else:
    plt.imshow(cv2.cvtColor(img_hough, cv2.COLOR_BGR2RGB))
plt.title("Círculos detectados y completados con Hough")
plt.axis("off")
plt.show()


MIN_AREA_cir = 3000  # cambiá este valor según necesites
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_hough, connectivity=8)



# Creamos una máscara vacía
mask_filtrada = np.zeros_like(img_hough)

# Recorremos las etiquetas (saltando la 0, que es fondo)
for label in range(1, num_labels):
    area = stats[label, cv2.CC_STAT_AREA]

    if area >= MIN_AREA_cir:
        # Copiamos el componente a la máscara
        mask_filtrada[labels == label] = 255




kernel_aper = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

img_open = cv2.morphologyEx(mask_filtrada, kernel=kernel_aper, op=cv2.MORPH_OPEN)

plt.figure(figsize=(6,6))
plt.imshow(img_open, cmap='gray')
plt.title("Componentes filtrados por área")
plt.axis("off")
plt.show()

areas = stats[1:, cv2.CC_STAT_AREA]  # todos menos el fondo
moneda_10 = []
moneda_50 = []
moneda_1 = []

for a in areas:
    if a < 75000:
        moneda_10.append(a)
    elif a > 95000:
        moneda_50.append(a)
    else:
        moneda_1.append(a)

print(f"Monedas de 10 centavos: {len(moneda_10)}")
print(f"Monedas de 50 centavos: {len(moneda_50)}")
print(f"Monedas de 1 peso: {len(moneda_1)}")

    

















