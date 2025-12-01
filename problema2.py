from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def segmentar_patente(img):
    imagen_color = img

    if imagen_color is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen")

    imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    # Detectamos los bordes con un filtro laplaciano
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img_laplaciano = cv2.filter2D(imagen_gris, -1, kernel)

    _, img_bin = cv2.threshold(
        img_laplaciano, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_bin, connectivity=8
    )

    min_area = float("inf")
    bbox_coords = None

    img_bin_color = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / float(h)
        area_bounding_box = w * h

        # Eliminamos los componentes que no cumplen con los criterios
        if aspect_ratio >= 1.5 and aspect_ratio <= 2.8 and area >= 200 and area <= 800:
            # Si cumple los criterios, verificar si es el de menor area hasta ahora
            if area_bounding_box < min_area:
                min_area = area_bounding_box
                bbox_coords = (x, y, w, h)

    if bbox_coords is not None:
        x, y, w, h = bbox_coords
        cv2.rectangle(img_bin_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_bin_color, cv2.COLOR_BGR2RGB))
    plt.title(f"Bounding box del componente con menor área ({min_area:.0f})")
    plt.axis("off")
    plt.show(block=True)

    imagen_crop = None

    # Con las coordenadas del bounding box de la patente, recortamos la imagen original
    if bbox_coords is not None:
        x, y, w, h = bbox_coords
        imagen_crop = imagen_color[y : y + h, x : x + w]

        plt.figure(figsize=(8, 4))
        plt.imshow(cv2.cvtColor(imagen_crop, cv2.COLOR_BGR2RGB))
        plt.title("Recorte (Crop) de la Placa Detectada")
        plt.axis("off")
        plt.show(block=True)
    else:
        print(
            "No se encontró ningún componente que cumpla con los criterios de filtrado."
        )

    imagen_crop_upscaled = None

    if imagen_crop is not None:
        # Como resultado tenemos una imagen muy pequeña del crop de la patente.
        # Para mejorar la segmentacion de los caracteres, reescalamos la imagen.
        scale_factor = 75

        imagen_crop_upscaled = cv2.resize(
            imagen_crop,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

    # Vamos a repetir el proceso de binarización con el crop de la patente reescalado

    crop_bin = np.array([])

    if imagen_crop_upscaled is not None:
        crop_gris = cv2.cvtColor(imagen_crop_upscaled, cv2.COLOR_BGR2GRAY)

        _, crop_bin = cv2.threshold(
            crop_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        plt.figure(figsize=(8, 4))
        plt.imshow(cv2.cvtColor(crop_bin, cv2.COLOR_BGR2RGB))
        plt.title("CROP BINARIZADO")
        plt.axis("off")
        plt.show(block=True)

    # Ahora que la imagen está reescalada, aplicamos una erosión para separar el fondo de los caracteres.
    morph = np.array([])

    if crop_bin is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (85, 85))
        morph = cv2.erode(crop_bin, kernel, iterations=1)

        plt.figure(figsize=(8, 8))
        plt.imshow(morph, cmap="gray")
        plt.title(f"Erosión Mínima (3x3, 1 iteración) Post-Reescalado")
        plt.axis("off")
        plt.show(block=True)

    # Ahora recortamos un margen exterior para eliminar posibles restos de fondo blanco conectados con los caracteres de la patente
    final_trim_margin = 70

    # Aseguramos que el recorte no resulte en una imagen vacia
    if (
        morph.shape[0] > 2 * final_trim_margin
        and morph.shape[1] > 2 * final_trim_margin
    ):
        morph_trimmed = morph[
            final_trim_margin:-final_trim_margin, final_trim_margin:-final_trim_margin
        ]
    else:
        print(
            f"El recorte final con margen={final_trim_margin} es demasiado grande. Usando la imagen sin recortar."
        )
        # Si el margen es demasiado grande, usa la imagen original
        morph_trimmed = morph.copy()

    plt.figure(figsize=(8, 8))
    plt.imshow(morph_trimmed, cmap="gray")
    plt.title(f"Resultado Final: Erosión + Recorte Exterior ({final_trim_margin}px)")
    plt.axis("off")
    plt.show(block=True)

    # Ahora vamos a segmentar los caracteres individuales dentro del crop de la patente
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        morph_trimmed, connectivity=8
    )

    # Obtenemos las dimensiones de la imagen para el filtrado relativo
    H, W = morph_trimmed.shape

    crop_bin_color = cv2.cvtColor(morph_trimmed, cv2.COLOR_GRAY2BGR)
    valid_bboxes = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / float(h)

        # Queremos que la altura sea significativa, por ejemplo, entre el 40% y el 100% de la altura total.
        min_char_height = H * 0.40
        max_char_height = H * 1.00

        # Descartar ruido muy pequeño. El área mínima debe ser > 500 píxeles.
        min_area_threshold = 500
        area_bounding_box = w * h

        if (
            h > w
            and area_bounding_box > min_area_threshold
            and h >= min_char_height
            and h <= max_char_height
        ):
            valid_bboxes.append((x, y, w, h))
            cv2.rectangle(crop_bin_color, (x, y), (x + w, y + h), (0, 255, 0), 3)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(crop_bin_color, cv2.COLOR_BGR2RGB))
    plt.title(f"Componentes Válidos Detectados: {len(valid_bboxes)}")
    plt.axis("off")
    plt.show(block=True)

    # Si existen 6 bounding boxes validos, procedemos a recortar y mostrar cada caracter individualmente
    # NOTA: El recorte se hace sobre el crop binarizado ya que es donde se visualizan mas claramente los caracteres
    if len(valid_bboxes) == 6:
        # Ordenar los bounding boxes de izquierda a derecha segun la coordenada x
        valid_bboxes.sort(key=lambda bbox: bbox[0])

        fig, axes = plt.subplots(1, 6, figsize=(15, 3))
        fig.suptitle("Segmentación Final de Caracteres", fontsize=16)

        # Vamos a definir un margen adicional para cada recorte ya que los caracteres pueden estar muy juntos con el borde del bounding box
        margin_pixels = 120

        # Obtener las dimensiones de la imagen binarizada reescalada para asegurarnos de que los recortes no se salgan de los límites.
        h_img, w_img = crop_bin.shape

        for i, (x, y, w, h) in enumerate(valid_bboxes):
            x_m = x - margin_pixels
            y_m = y - margin_pixels
            w_m = w + (2 * margin_pixels)
            h_m = h + (2 * margin_pixels)

            # Coordenadas dentro de los límites de la imagen
            x_final = max(0, x_m)
            y_final = max(0, y_m)
            
            # Finales dentro de los límites de la imagen
            x_final_end = min(w_img, x_m + w_m)
            y_final_end = min(h_img, y_m + h_m)
            
            caracter_crop = crop_bin[y_final:y_final_end, x_final:x_final_end]

            axes[i].imshow(caracter_crop, cmap="gray")
            axes[i].set_title(f"Carácter {i+1}")
            axes[i].axis("off")

        plt.show(block=True)
    else:
        print("NO SE ENCONTRÓ LA PATENTE")

    return valid_bboxes

try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

imagenes_path = BASE_DIR / "imagenes"
patentes = sorted(p for p in imagenes_path.glob("img*.png"))

imagen = cv2.imread(str(patentes[7]))

segmentar_patente(imagen)
