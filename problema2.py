from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def binarizar_otsu(imagen):
    """
    Binariza una imagen usando el método de Otsu.

    Args:
        imagen: Imagen en escala de grises.

    Returns:
        Imagen binarizada.
    """
    _, img_bin = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_bin


def mostrar_imagen(imagen, titulo="Imagen", figsize=(8, 8), cmap=None, block=False):
    """
    Muestra una imagen con matplotlib.

    Args:
        imagen: Imagen a mostrar (BGR o escala de grises).
        titulo: Título de la figura.
        figsize: Tamaño de la figura.
        cmap: Colormap (None para BGR, 'gray' para escala de grises).
        block: Si bloquea la ejecución hasta cerrar la figura.
    """
    plt.figure(figsize=figsize)

    if cmap is None and len(imagen.shape) == 3:
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(imagen, cmap=cmap or "gray")

    plt.title(titulo)
    plt.axis("off")
    plt.show(block=block)


def dibujar_bounding_boxes(imagen, bboxes, color=(0, 255, 0), grosor=2):
    """
    Dibuja bounding boxes sobre una imagen.

    Args:
        imagen: Imagen base (se crea una copia en color si es necesario).
        bboxes: Lista de tuplas (x, y, w, h).
        color: Color BGR para los rectángulos.
        grosor: Grosor de las líneas.

    Returns:
        Imagen con los bounding boxes dibujados.
    """
    if len(imagen.shape) == 2:
        img_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        img_color = imagen.copy()

    for x, y, w, h in bboxes:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), color, grosor)

    return img_color


def segmentar_patente(
    img,
    mostrar_pasos=True,
    scale_factor=75,
    erosion_kernel=(85, 85),
    trim_margin=70,
    char_margin=120,
    num_caracteres_esperados=6,
):
    """
    Segmenta una patente en una imagen y extrae sus caracteres.

    Args:
        img: Imagen de entrada en formato BGR.
        mostrar_pasos: Si se muestran los pasos intermedios (default: True).
        scale_factor: Factor de escala para el upscaling (default: 75).
        erosion_kernel: Tamaño del kernel de erosión (default: (85, 85)).
        trim_margin: Margen para recortar bordes (default: 70).
        char_margin: Margen para recortar caracteres (default: 120).
        num_caracteres_esperados: Número de caracteres esperados en la patente (default: 6).

    Returns:
        Tupla (valid_bboxes, crop_bin) donde:
            - valid_bboxes: Lista de bounding boxes de los caracteres detectados.
            - crop_bin: Imagen binarizada del crop de la patente reescalada.
    """
    if img is None:
        raise FileNotFoundError("No se pudo cargar la imagen")

    imagen_color = img
    imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    # Detectamos los bordes con un filtro laplaciano
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img_laplaciano = cv2.filter2D(imagen_gris, -1, kernel)
    img_bin = binarizar_otsu(img_laplaciano)

    # Encontrar la patente usando componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_bin, connectivity=8
    )

    # Eliminamos los componentes que no cumplen con los criterios
    # Si cumple los criterios, verificar si es el de menor area hasta ahora
    min_bbox_area = float("inf")
    bbox_patente = None

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / float(h)
        area_bounding_box = w * h

        if 1.5 <= aspect_ratio <= 2.8 and 200 <= area <= 800:
            if area_bounding_box < min_bbox_area:
                min_bbox_area = area_bounding_box
                bbox_patente = (x, y, w, h)

    min_area = min_bbox_area if bbox_patente else None

    if mostrar_pasos and bbox_patente is not None:
        img_con_bbox = dibujar_bounding_boxes(
            img_bin, [bbox_patente], color=(0, 0, 255)
        )
        mostrar_imagen(
            img_con_bbox, f"Bounding box del componente con menor área ({min_area:.0f})"
        )

    if bbox_patente is None:
        print(
            "No se encontró ningún componente que cumpla con los criterios de filtrado."
        )
        return [], None

    # Con las coordenadas del bounding box de la patente, recortamos la imagen original
    x, y, w, h = bbox_patente
    imagen_crop = imagen_color[y : y + h, x : x + w]

    if mostrar_pasos:
        mostrar_imagen(
            imagen_crop, "Recorte (Crop) de la Placa Detectada", figsize=(8, 4)
        )

    # Como resultado tenemos una imagen muy pequeña del crop de la patente.
    # Para mejorar la segmentacion de los caracteres, reescalamos la imagen.
    imagen_crop_upscaled = cv2.resize(
        imagen_crop,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    # Vamos a repetir el proceso de binarización con el crop de la patente reescalado
    crop_gris = cv2.cvtColor(imagen_crop_upscaled, cv2.COLOR_BGR2GRAY)
    crop_bin = binarizar_otsu(crop_gris)

    if mostrar_pasos:
        mostrar_imagen(crop_bin, "CROP BINARIZADO", figsize=(8, 4), cmap="gray")

    # Ahora que la imagen está reescalada, aplicamos una erosión para separar el fondo de los caracteres.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_kernel)
    morph = cv2.erode(crop_bin, kernel, iterations=1)

    if mostrar_pasos:
        mostrar_imagen(
            morph, f"Erosión ({erosion_kernel}) Post-Reescalado", cmap="gray"
        )

    # Ahora recortamos un margen exterior para eliminar posibles restos de fondo blanco conectados con los caracteres de la patente
    if morph.shape[0] > 2 * trim_margin and morph.shape[1] > 2 * trim_margin:
        morph_trimmed = morph[trim_margin:-trim_margin, trim_margin:-trim_margin]
    else:
        print(
            f"El recorte con margen={trim_margin} es demasiado grande. Usando imagen sin recortar."
        )
        morph_trimmed = morph.copy()

    if mostrar_pasos:
        mostrar_imagen(
            morph_trimmed,
            f"Resultado Final: Erosión + Recorte Exterior ({trim_margin}px)",
            cmap="gray",
        )

    # Ahora vamos a segmentar los caracteres individuales dentro del crop de la patente
    # Obtenemos las dimensiones de la imagen para el filtrado relativo
    # Queremos que la altura sea significativa, por ejemplo, entre el 40% y el 100% de la altura total.
    # Descartar ruido muy pequeño. El área mínima debe ser > 500 píxeles.
    num_labels, labels, stats_chars, centroids = cv2.connectedComponentsWithStats(
        morph_trimmed, connectivity=8
    )
    H, W = morph_trimmed.shape

    min_char_height = H * 0.40
    max_char_height = H * 1.00
    min_area_threshold = 500

    valid_bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats_chars[i]
        area_bbox = w * h
        if (
            h > w
            and area_bbox > min_area_threshold
            and min_char_height <= h <= max_char_height
        ):
            valid_bboxes.append((x, y, w, h))

    if mostrar_pasos:
        img_con_chars = dibujar_bounding_boxes(
            morph_trimmed, valid_bboxes, color=(0, 255, 0), grosor=3
        )
        mostrar_imagen(
            img_con_chars, f"Componentes Válidos Detectados: {len(valid_bboxes)}"
        )

    # Si existen los bounding boxes validos esperados, procedemos a recortar y mostrar cada caracter individualmente
    # NOTA: El recorte se hace sobre el crop binarizado ya que es donde se visualizan mas claramente los caracteres
    if len(valid_bboxes) == num_caracteres_esperados:
        # Ordenar los bounding boxes de izquierda a derecha segun la coordenada x
        valid_bboxes.sort(key=lambda bbox: bbox[0])

        # Mostrar los caracteres segmentados en una fila
        n_chars = len(valid_bboxes)
        fig, axes = plt.subplots(1, n_chars, figsize=(15, 3))
        fig.suptitle("Segmentación Final de Caracteres", fontsize=16)

        # Asegurar que axes sea iterable incluso con un solo caracter
        if n_chars == 1:
            axes = [axes]

        h_img, w_img = crop_bin.shape[:2]
        for i, (bx, by, bw, bh) in enumerate(valid_bboxes):
            # Recortar con margen adicional respetando los límites de la imagen
            x_inicio = max(0, bx - char_margin)
            y_inicio = max(0, by - char_margin)
            x_fin = min(w_img, bx + bw + char_margin)
            y_fin = min(h_img, by + bh + char_margin)
            caracter_crop = crop_bin[y_inicio:y_fin, x_inicio:x_fin]

            axes[i].imshow(caracter_crop, cmap="gray")
            axes[i].set_title(f"Carácter {i + 1}")
            axes[i].axis("off")

        plt.show(block=False)
    else:
        print(
            f"NO SE ENCONTRÓ LA PATENTE (se encontraron {len(valid_bboxes)} caracteres, se esperaban {num_caracteres_esperados})"
        )

    return valid_bboxes, crop_bin


if __name__ == "__main__":
    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path.cwd()

    imagenes_path = BASE_DIR / "imagenes"
    patentes = sorted(p for p in imagenes_path.glob("img*.png"))

    for i, p in enumerate(patentes):
        imagen = cv2.imread(str(p))

        # Llamada básica con visualización
        bboxes, crop_bin = segmentar_patente(imagen, mostrar_pasos=False)

    # Ejemplo de uso con parámetros personalizados:
    # bboxes, crop_bin = segmentar_patente(
    #     imagen,
    #     mostrar_pasos=False,          # Sin visualización
    #     scale_factor=50,              # Escala diferente
    #     erosion_kernel=(70, 70),      # Kernel de erosión más pequeño
    #     trim_margin=50,               # Margen de recorte diferente
    #     char_margin=100,              # Margen para caracteres
    #     num_caracteres_esperados=7    # Para patentes con diferente formato
    # )
