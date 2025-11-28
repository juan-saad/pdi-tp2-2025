from email.mime import image
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
def graficar_connected_components(img_binaria):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_binaria, connectivity=8
    )

    img_bin_color = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)

    for i in range(1, num_labels):  # Empieza en 1 para saltar el fondo
        x, y, w, h, area = stats[i]
        aspect_ratio = w / float(h)
        if aspect_ratio >= 1.5 and aspect_ratio <= 2.8 and area >= 200 and area <= 800:
            cv2.rectangle(img_bin_color, (x, y), (x + w, y + h), (0, 0, 255), 1)


    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_bin_color, cv2.COLOR_BGR2RGB))
    plt.title("Bounding box de cada componente en la imagen binaria")
    plt.axis("off")
    plt.show(block=True)
'''

def graficar_connected_components(img_binaria):
    # 1. Encontrar los componentes conectados y sus estadísticas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_binaria, connectivity=8
    )

    # 2. Inicializar variables para encontrar la caja de menor área
    min_area = float('inf')
    best_bbox = None  # Almacenará (x, y, w, h) del mejor componente
    
    # img_bin_color se usa para la visualización (copia de la binaria en BGR)
    img_bin_color = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)

    # 3. Iterar sobre todos los componentes (excepto el fondo, índice 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / float(h)
        area_bounding_box = w * h
        
        # 4. Verificar los criterios de filtrado
        # Criterios actuales: aspect_ratio >= 1.5 and aspect_ratio <= 2.8 and area >= 200 and area <= 800
        if aspect_ratio >= 1.5 and aspect_ratio <= 2.8 and area >= 200 and area <= 800:
            # 5. Si cumple los criterios, verificar si es el de menor área hasta ahora
            if area_bounding_box < min_area:
                min_area = area_bounding_box
                best_bbox = (x, y, w, h)
                
    # 6. Dibujar el bounding box SOLO del componente de menor área (si se encontró alguno)
    if best_bbox is not None:
        x, y, w, h = best_bbox
        # Dibujamos el rectángulo rojo sobre la imagen binaria en color
        cv2.rectangle(img_bin_color, (x, y), (x + w, y + h), (0, 0, 255), 2) # Grosor 2 para más visibilidad


    # 7. Mostrar el resultado
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_bin_color, cv2.COLOR_BGR2RGB))
    plt.title(f"Bounding box del componente con menor área ({min_area:.0f})")
    plt.axis("off")
    plt.show(block=True)

    return best_bbox    


try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

imagenes_path = BASE_DIR / "imagenes"
patentes = sorted(p for p in imagenes_path.glob("img*.png"))
'''
for imagen_a_procesar_path in patentes:

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

    # kernel_alt = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    kernel_alt = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    img_sharp = cv2.filter2D(imagen_gris, -1, kernel_alt)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    # axes[0].imshow(img_sharp, cmap="gray")
    # axes[0].set_title("Imagen Sharpened")
    # axes[1].imshow(imagen_gris, cmap="gray")
    # axes[1].set_title("Imagen en escala de grises")
    # fig.suptitle("Comparación Imagen Sharpened", fontsize=14)
    # plt.tight_layout()
    # plt.show(block=False)

    _, img_bin = cv2.threshold(img_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    # axes[0].imshow(img_bin, cmap="gray")
    # axes[0].set_title("Binarización Otsu")
    # axes[1].imshow(img_sharp, cmap="gray")
    # axes[1].set_title("Imagen en escala de grises")

    # fig.suptitle("Comparación de Binarización Otsu", fontsize=14)
    # plt.tight_layout()
    # plt.show(block=False)

    graficar_connected_components(img_bin)
'''

#-------------------------------------------------------------------------------------------------------------


imagen_color = cv2.imread(str(patentes[10]))

if imagen_color is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen")

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

# kernel_alt = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

kernel_alt = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

img_sharp = cv2.filter2D(imagen_gris, -1, kernel_alt)

# fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
# axes[0].imshow(img_sharp, cmap="gray")
# axes[0].set_title("Imagen Sharpened")
# axes[1].imshow(imagen_gris, cmap="gray")
# axes[1].set_title("Imagen en escala de grises")
# fig.suptitle("Comparación Imagen Sharpened", fontsize=14)
# plt.tight_layout()
# plt.show(block=False)

_, img_bin = cv2.threshold(img_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
# axes[0].imshow(img_bin, cmap="gray")
# axes[0].set_title("Binarización Otsu")
# axes[1].imshow(img_sharp, cmap="gray")
# axes[1].set_title("Imagen en escala de grises")

# fig.suptitle("Comparación de Binarización Otsu", fontsize=14)
# plt.tight_layout()
# plt.show(block=False)

bbox_coords = graficar_connected_components(img_bin)

#------------------------------RECORTAMOS EL BOUNDING BOX DE LA PATENTE-----------------------------------------------------------------

if bbox_coords is not None:
    # Desempaquetar las coordenadas
    x, y, w, h = bbox_coords
    
    # La indexación de matrices en NumPy/OpenCV es [filas (y), columnas (x)]
    # Filas: desde y hasta y + h
    # Columnas: desde x hasta x + w
    imagen_crop = imagen_color[y:y + h, x:x + w]

    # -------------------------------------------------------------
    ## 🖼️ Mostrar el Recorte (Crop)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    # OpenCV lee BGR, Matplotlib necesita RGB
    plt.imshow(cv2.cvtColor(imagen_crop, cv2.COLOR_BGR2RGB)) 
    plt.title("Recorte (Crop) de la Placa Detectada")
    plt.axis("off")
    plt.show(block=True)

else:
    print("No se encontró ningún componente que cumpla con los criterios de filtrado.") 

#---------------------------PROCESAMOS EL CROP DE LA PATENTE---------------------------------------------------------------------------

# La pasamos a escala de grises
crop_gris = cv2.cvtColor(imagen_crop, cv2.COLOR_BGR2GRAY)

# La binarizamos
_, crop_bin = cv2.threshold(crop_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(8, 4))
# OpenCV lee BGR, Matplotlib necesita RGB
plt.imshow(cv2.cvtColor(crop_bin, cv2.COLOR_BGR2RGB)) 
plt.title("CROP BINARIZADO")
plt.axis("off")
plt.show(block=True)



#-----------------------SEGMENTAR LOS CARACTERES DEL CROP DE LA PATENTE------------------------------------------

def caracteres_connected_components(img_binaria):
    # 1. Encontrar los componentes conectados y sus estadísticas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_binaria, connectivity=8
    )

    # 2. Inicializar variables para encontrar la caja de menor área
    #min_area = float('inf')
    #best_bbox = None  # Almacenará (x, y, w, h) del mejor componente
    
    # img_bin_color se usa para la visualización (copia de la binaria en BGR)
    crop_bin_color = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)

    # 3. Iterar sobre todos los componentes (excepto el fondo, índice 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        #aspect_ratio = w / float(h)
        #area_bounding_box = w * h
        
        cv2.rectangle(crop_bin_color, (x, y), (x + w, y + h), (0, 0, 255), 1) # Grosor 2 para más visibilidad


    # 7. Mostrar el resultado
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(crop_bin_color, cv2.COLOR_BGR2RGB))
    plt.title(f"Bounding box dentro de la patente")
    plt.axis("off")
    plt.show(block=True)

caracteres_connected_components(crop_bin)
