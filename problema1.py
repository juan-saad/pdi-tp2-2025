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
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Top-hat
kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
img_tophat = cv2.morphologyEx(img_gray, kernel=kernel_tophat, op=cv2.MORPH_TOPHAT)

# Gradient
kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
img_grad = cv2.morphologyEx(img_tophat, cv2.MORPH_GRADIENT, kernel_grad)

# Binarización Otsu
_ , img_bin = cv2.threshold(img_grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_bin = cv2.GaussianBlur(img_bin, (5, 5), 2)



MIN_AREA = 3000     # umbral de área mínimo

# contornos
contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# filtramos contornos por área
contours_filtrados = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

# imagen mismo tamaño que la original, para dibujar contornos
mask_contornos = np.zeros_like(img_bin)

# dibujo de contornos filtrados
cv2.drawContours(mask_contornos, contours_filtrados, -1, 255, -1)

# aplico canny para diferenciar bordes
contornos_canny = cv2.Canny(mask_contornos, 50, 150)


# detección de círculos con Hough
circles = cv2.HoughCircles(
    contornos_canny,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=300,
    param1=50,
    param2=40,
    minRadius=10,
    maxRadius=200
)

img_hough = contornos_canny.copy()  

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0,:]:
        cv2.circle(img_hough, (x, y), r, (255,255,255), -1)  

plt.figure(figsize=(6,6))
if len(img_hough.shape) == 2:
    plt.imshow(img_hough, cmap='gray')
else:
    plt.imshow(cv2.cvtColor(img_hough, cv2.COLOR_BGR2RGB))
plt.title("Círculos detectados y completados con Hough")
plt.axis("off")
plt.show()



num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_hough, connectivity=8)

mask_filtrada = np.zeros_like(img_hough)

# Recorremos las etiquetas (saltando la 0, que es fondo)
for label in range(1, num_labels):
    area = stats[label, cv2.CC_STAT_AREA]

    if area >= MIN_AREA:
        mask_filtrada[labels == label] = 255

plt.figure(figsize=(6,6))
plt.imshow(mask_filtrada, cmap='gray')
plt.title("Componentes filtrados por área")
plt.axis("off")
plt.show()



_ , _ , stats_final, _ = cv2.connectedComponentsWithStats(mask_filtrada, connectivity=8)



areas = stats_final[1:, cv2.CC_STAT_AREA]  # todos menos el fondo
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

    

















