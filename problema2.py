from email.mime import image
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

imagenes_path = BASE_DIR / "imagenes"
patentes = sorted(p for p in imagenes_path.glob("img*.png"))


imagen_a_procesar_path = patentes[4]
print(f"Cargando imagen: {imagen_a_procesar_path.name}")

imagen_color = cv2.imread(str(imagen_a_procesar_path))

if imagen_color is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {imagen_a_procesar_path}")

imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

# kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
# img_tophat = cv2.morphologyEx(imagen_gris, kernel=kernel_tophat, op=cv2.MORPH_TOPHAT)

# fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# # Mostrar img_tophat
# axes[0, 0].imshow(img_tophat, cmap="gray")
# axes[0, 0].set_title("Imagen Tophat")
# axes[0, 0].axis("off")

# # Histograma de img_tophat
# axes[1, 0].hist(img_tophat.ravel(), bins=256, range=(0, 256), color="black")
# axes[1, 0].set_title("Histograma de Tophat")
# axes[1, 0].set_xlim([0, 256])

# # Mostrar imagen_gris
# axes[0, 1].imshow(imagen_gris, cmap="gray")
# axes[0, 1].set_title("Imagen en Escala de Grises")
# axes[0, 1].axis("off")

# # Histograma de imagen_gris
# axes[1, 1].hist(imagen_gris.ravel(), bins=256, range=(0, 256), color="black")
# axes[1, 1].set_title("Histograma de Escala de Grises")
# axes[1, 1].set_xlim([0, 256])

# plt.show(block=False)

# kernel_clasico = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

kernel_alt = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

img_sharp = cv2.filter2D(imagen_gris, -1, kernel_alt)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
axes[0].imshow(img_sharp, cmap="gray")
axes[0].set_title("Imagen Sharpened")
axes[1].imshow(imagen_gris, cmap="gray")
axes[1].set_title("Imagen en escala de grises")
fig.suptitle("Comparación Imagen Sharpened", fontsize=14)
plt.tight_layout()
plt.show(block=False)

_, img_bin = cv2.threshold(img_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
axes[0].imshow(img_bin, cmap="gray")
axes[0].set_title("Binarización Otsu")
axes[1].imshow(img_sharp, cmap="gray")
axes[1].set_title("Imagen en escala de grises")

fig.suptitle("Comparación de Binarización Otsu", fontsize=14)
plt.tight_layout()
plt.show(block=False)

img_close = cv2.morphologyEx(
    img_bin, cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
)
img_open_close = cv2.morphologyEx(
    img_close, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
axes[0].imshow(img_open_close, cmap="gray")
axes[0].set_title("Imagen después de Apertura Morfológica")
axes[1].imshow(img_close, cmap="gray")
axes[1].set_title("Imagen después de Clausura Morfológica")

fig.suptitle("Comparación de Apertura y Cierre Morfológico", fontsize=14)
plt.tight_layout()
plt.show(block=False)

contornos, _ = cv2.findContours(img_open_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask_contornos = np.zeros_like(imagen_color)
cv2.drawContours(mask_contornos, contornos, -1, (0, 255, 0), 1)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(mask_contornos, cv2.COLOR_BGR2RGB))
plt.title("Bounding box monedas y contornos dados superpuestos")
plt.axis("off")
plt.show(block=False)
