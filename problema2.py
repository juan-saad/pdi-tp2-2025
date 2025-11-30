from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calcular_umbral_areas(imagenes_binarias, percentil=95):
    areas = []

    # Asegurarse de que sea una lista
    if not isinstance(imagenes_binarias, list):
        imagenes_binarias = [imagenes_binarias]

    for img_binaria in imagenes_binarias:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            img_binaria, connectivity=8
        )
        areas.extend(stats[1:, cv2.CC_STAT_AREA])

    if areas:
        return np.percentile(areas, percentil)
    else:
        return 0


def calcular_umbral_aspect_ratio(
    imagenes_binarias, umbral_area, percentil_min=15, percentil_max=85
):
    aspect_ratios = []

    # Asegurarse de que sea una lista
    if not isinstance(imagenes_binarias, list):
        imagenes_binarias = [imagenes_binarias]

    for img_binaria in imagenes_binarias:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            img_binaria, connectivity=8
        )

        for i in range(1, num_labels):  # Saltar el fondo
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= umbral_area and h > 0:  # Evitar división por cero
                aspect_ratio = w / float(h)
                aspect_ratios.append(aspect_ratio)

    if aspect_ratios:
        # Calcular los percentiles para definir el rango
        aspect_ratio_min = np.percentile(aspect_ratios, percentil_min)
        aspect_ratio_max = np.percentile(aspect_ratios, percentil_max)
        return (aspect_ratio_min, aspect_ratio_max)
    else:
        return (0, 0)


def filtrar_por_area_aspect_ratio(img_binaria, umbral_area, umbral_aspect_ratio):
    """Descarta componentes conectados con área menor al umbral."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        img_binaria, connectivity=8
    )

    img_filtrada = np.zeros_like(img_binaria)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        aspect_ratio = w / float(h) if h > 0 else 0

        if (
            area >= umbral_area
            and umbral_aspect_ratio[0] < aspect_ratio <= umbral_aspect_ratio[1]
        ):
            img_filtrada[labels == i] = 255

    return img_filtrada


def graficar_connected_components(img_binaria, umbral_aspect_ratio, umbral_area):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_binaria, connectivity=8
    )

    img_bin_color = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)

    for i in range(1, num_labels):  # Empieza en 1 para saltar el fondo
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        aspect_ratio = w / float(h) if h > 0 else 0

        # Verificar componentes que cumplen con el área y el aspect ratio
        if (
            area >= umbral_area
            and umbral_aspect_ratio[0] < aspect_ratio <= umbral_aspect_ratio[1]
        ):
            # Dibujar el bounding box en rojo
            cv2.rectangle(img_bin_color, (x, y), (x + w, y + h), (0, 0, 255), 1)

            # Agregar la etiqueta en el centroide
            etiqueta = f"{i}-{area}-{aspect_ratio:.2f}"
            cv2.putText(
                img_bin_color,
                etiqueta,
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_bin_color, cv2.COLOR_BGR2RGB))
    plt.title("Bounding boxes y etiquetas de componentes conectados")
    plt.axis("off")
    plt.show(block=False)


def limpiar_bordes(img_binaria):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        img_binaria, connectivity=8
    )

    # Crear una máscara binaria inicializada en negro
    img_sin_bordes = np.zeros_like(img_binaria)

    alto, ancho = img_binaria.shape

    for i in range(1, num_labels):  # Saltar el fondo (etiqueta 0)
        x, y, w, h, area = stats[i]

        # Verificar si el componente toca algún borde
        if x > 0 and y > 0 and (x + w) < ancho and (y + h) < alto:
            # Si no toca el borde, mantenerlo en la nueva máscara
            img_sin_bordes[labels == i] = 255

    return img_sin_bordes

def calcular_contornos(img_binaria, umbral_area):
    contours, hierarchy = cv2.findContours(
        img_binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    contornos_finales = []

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        
        for i, cnt in enumerate(contours):
            # Obtengo el primer hijo del contorno actual
            child = hierarchy[i][2]
            
            # 1. Calcular el Área
            area = cv2.contourArea(cnt)
            
            # Condición A: Debe ser una figura cerrada (Tener hijo/hueco)
            es_cerrado = (child != -1)
            
            # Condición B: Debe ser más grande que el ruido/líneas
            es_grande = (area > umbral_area)
            
            # Solo si cumple AMBAS, lo guardamos
            if es_cerrado and es_grande:
                print(f"Contorno {i}: Área = {area}, Cerrado = {es_cerrado}, Grande = {es_grande}")
                contornos_finales.append(cnt)
    
    return contornos_finales


try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

imagenes_path = BASE_DIR / "imagenes"
patentes = sorted(p for p in imagenes_path.glob("img*.png"))

imagenes_binarias = []

for imagen_a_procesar_path in patentes:
    imagen_color = cv2.imread(str(imagen_a_procesar_path))

    if imagen_color is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen: {imagen_a_procesar_path}"
        )

    imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
    kernel_alt = -1 * np.ones((3, 3))
    kernel_alt[1, 1] = 8
    img_sharp = cv2.filter2D(
        imagen_gris, -1, kernel_alt, borderType=cv2.BORDER_CONSTANT
    )
    _, img_bin = cv2.threshold(img_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    imagenes_binarias.append(img_bin)

umbral_area = calcular_umbral_areas(imagenes_binarias)
umbral_aspect_ratio = calcular_umbral_aspect_ratio(
    imagenes_binarias, umbral_area, percentil_min=10, percentil_max=85
)

imagenes_filtradas = []

# Aplicar el filtro a cada imagen
for img_bin in imagenes_binarias:
    img_filtrada = filtrar_por_area_aspect_ratio(
        img_bin, umbral_area, umbral_aspect_ratio
    )
    imagenes_filtradas.append(img_filtrada)
    plt.figure()
    plt.imshow(img_filtrada, cmap="gray")
    plt.title(f"Imagenes filtradas")
    plt.axis("off")

plt.show(block=False)

img = imagenes_filtradas[5]

nuevo_umbral_area = calcular_umbral_areas(img, percentil=95)
nuevo_umbral_aspect_ratio = calcular_umbral_aspect_ratio(
    img, nuevo_umbral_area, percentil_min=10, percentil_max=85
)

# Aplicar la función a la imagen binaria
img_sin_bordes = limpiar_bordes(img)
contornos_filtrados = calcular_contornos(img_sin_bordes, nuevo_umbral_area)

# Dibujar los contornos filtrados en una imagen nueva
img_contornos = np.zeros_like(img_sin_bordes)
cv2.drawContours(img_contornos, contornos_filtrados, -1, (255), thickness=1)

graficar_connected_components(img_contornos, nuevo_umbral_aspect_ratio, nuevo_umbral_area)








# for img in imagenes_filtradas:
#     nuevo_umbral_area = calcular_umbral_areas(img, percentil=95)
#     nuevo_umbral_aspect_ratio = calcular_umbral_aspect_ratio(
#         img, nuevo_umbral_area, percentil_min=10, percentil_max=85
#     )

#     img_sin_bordes = limpiar_bordes(img)
#     contornos_filtrados = calcular_contornos(img_sin_bordes, nuevo_umbral_area)

#     # Dibujar los contornos filtrados en una imagen nueva
#     img_contornos = np.zeros_like(img_sin_bordes)
#     cv2.drawContours(img_contornos, contornos_filtrados, -1, (255), thickness=1)
#     plt.figure()
#     plt.imshow(img_contornos, cmap="gray")
#     plt.title("Contornos filtrados")
#     plt.axis("off")
    
# plt.show(block=False)
