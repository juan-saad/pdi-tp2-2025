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




AmB = img_gray.copy()
AmB[mask_filtrada>0]=0
plt.imshow(AmB, cmap='gray')
plt.title("Contornos detectados")
plt.axis("off")
plt.show()

# bordes = cv2.Canny(AmB, 50, 150)
# plt.imshow(bordes, cmap='gray')
# plt.title("Bordes de la imagen original sin monedas")
# plt.axis("off")

# Top-hat
kernel_tophat_dados = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
img_tophat_dados = cv2.morphologyEx(AmB, kernel=kernel_tophat_dados, op=cv2.MORPH_TOPHAT)
plt.imshow(img_tophat_dados, cmap='gray')
plt.title("Contornos detectados")
plt.axis("off")
plt.show()
# Gradient
kernel_grad_dados = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
img_grad_dados = cv2.morphologyEx(img_tophat_dados, cv2.MORPH_GRADIENT, kernel_grad_dados)
plt.imshow(img_grad_dados, cmap='gray')
plt.title("Contornos detectados")
plt.axis("off")
plt.show()
# Binarización Otsu
_ , img_bin_dados = cv2.threshold(img_grad_dados, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_bin_dados = cv2.GaussianBlur(img_bin_dados, (5, 5), 2)
plt.imshow(img_bin_dados, cmap='gray')
plt.title("Contornos detectados")
plt.axis("off")
plt.show()

dados_canny = cv2.Canny(AmB, 50, 150)
plt.imshow(dados_canny, cmap='gray')
plt.title("Contornos detectados")
plt.axis("off")
plt.show()

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dados_canny, connectivity=8)

img_etiquetas = cv2.cvtColor(dados_canny, cv2.COLOR_GRAY2BGR)

for label in range(1, num_labels):
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]
    cv2.rectangle(img_etiquetas, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cx, cy = int(centroids[label][0]), int(centroids[label][1])
    cv2.putText(img_etiquetas, str(label), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

plt.figure(figsize=(6,6))
plt.imshow(img_etiquetas)
plt.title("Objetos detectados en dados_canny con etiquetas")
plt.axis("off")
plt.show()

MIN_AREA_DADOS = 180  # Ajusta este valor según el tamaño mínimo que quieras conservar
MAX_AREA_DADOS = 250
mask_dados_filtrados = np.zeros_like(dados_canny)

for label in range(1, num_labels):
    area = stats[label, cv2.CC_STAT_AREA]
    if area >= MIN_AREA_DADOS and area <= MAX_AREA_DADOS:
        mask_dados_filtrados[labels == label] = 255
    

plt.figure(figsize=(6,6))
plt.imshow(mask_dados_filtrados, cmap='gray')
plt.title("Dados filtrados por área mínima")
plt.axis("off")
plt.show()



circulos_dados = cv2.HoughCircles(
    mask_dados_filtrados,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=5,
    param1=50,
    param2=20,
    minRadius=20,
    maxRadius=30
)

img_daditos = mask_dados_filtrados.copy()  

if circulos_dados is not None:
    circulos_dados = np.uint16(np.around(circulos_dados))
    for x, y, r in circulos_dados[0,:]:
        cv2.circle(img_daditos, (x, y), r, (255,255,255), -1)  

plt.figure(figsize=(6,6))
if len(img_daditos.shape) == 2:
    plt.imshow(img_daditos, cmap='gray')
else:
    plt.imshow(cv2.cvtColor(img_daditos, cv2.COLOR_BGR2RGB))
plt.title("Círculos detectados y completados con Hough")
plt.axis("off")
plt.show()


area_filtro_pips = 500
num_labels_final_dados , labels_final_dados , stats_final_dados, centroids_final_dados = cv2.connectedComponentsWithStats(img_daditos, connectivity=8)

mask_pips_filtrados = np.zeros_like(img_daditos)

for label in range(1, num_labels_final_dados):
    area = stats_final_dados[label, cv2.CC_STAT_AREA]
    if area >= area_filtro_pips:
        mask_pips_filtrados[labels_final_dados == label] = 255

plt.figure(figsize=(6,6))
plt.imshow(mask_pips_filtrados, cmap='gray')
plt.title("Pips filtrados por área mínima")
plt.axis("off")
plt.show()

# calcular la cantidad de dados segundo la distancia entre los centroides de los objetos








