from email.mime import image
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------FUNCIÓN PARA SEGMENTAR LA PATENTE Y SUS 6 CARACTERES-----------------------------

def segmentar_patente(img):

    #imagen_color = cv2.imread(img)
    imagen_color = img

    if imagen_color is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen")

    imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    kernel_alt = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    img_sharp = cv2.filter2D(imagen_gris, -1, kernel_alt)

    _, img_bin = cv2.threshold(img_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 1. Encontrar los componentes conectados y sus estadísticas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_bin, connectivity=8
    )

    # 2. Inicializar variables para encontrar la caja de menor área
    min_area = float('inf')
    best_bbox = None  # Almacenará (x, y, w, h) del mejor componente
    
    # img_bin_color se usa para la visualización (copia de la binaria en BGR)
    img_bin_color = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

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

    # ---------GUARDAMOS EL BOUNDING BOX DE LA PATENTE EN UNA VARIABLE PARA LUEGO TRABAJAR CON SUS COORDNADAS---------

    bbox_coords = best_bbox

    #----------------USAMOS LAS COORDENADAS DEL BOUNDING BOX DE LA PATENTE PARA RECORTAR LA IMAGEN ORIGINAL--------------

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

    #-------------------------------REESCALAMIENO DE LA IMAGEN RECORTADA-------------------------------------------
    # 1. Definir el factor de escalado
    scale_factor = 75
    #scale_factor = 170

    # 2. Reescalar la imagen de la patente A COLOR (imagen_crop)
    # Usamos INTER_CUBIC o INTER_LANCZOS4 para mejor calidad al aumentar el tamaño.
    imagen_crop_upscaled = cv2.resize(
        imagen_crop, 
        None, 
        fx=scale_factor, 
        fy=scale_factor, 
        interpolation=cv2.INTER_CUBIC
    )

    #---------------------------PROCESAMOS EL CROP DE LA PATENTE---------------------------------------------------------------------------

    # La pasamos a escala de grises
    crop_gris = cv2.cvtColor(imagen_crop_upscaled, cv2.COLOR_BGR2GRAY)

    # La binarizamos
    _, crop_bin = cv2.threshold(crop_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.figure(figsize=(8, 4))
    # OpenCV lee BGR, Matplotlib necesita RGB
    plt.imshow(cv2.cvtColor(crop_bin, cv2.COLOR_BGR2RGB)) 
    plt.title("CROP BINARIZADO")
    plt.axis("off")
    plt.show(block=True)

    #------------------------------------------APLICAMOS MORFOLOGÍA-CAMBIOS ----------------------------------

    # Ahora que la imagen está reescalada, aplicamos una erosión
    # para romper la conexión del fondo con los caracteres sin deformarlos.

    # 1. Definir un kernel pequeño (3x3 es el estándar para la erosión mínima)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (85, 85))

    # 2. Aplicar la Erosión con una sola iteración
    morph = cv2.erode(crop_bin, kernel, iterations=1) # El resultado se guarda en la varaible 'morph'

    plt.figure(figsize=(8, 8))
    plt.imshow(morph, cmap="gray")
    plt.title(f"Erosión Mínima (3x3, 1 iteración) Post-Reescalado")
    plt.axis("off")
    plt.show(block=True)

    #-----------------------------------RECORTAMOS NUEVAMENTE LA IMAGEN PARA DESCARTAR LOS BORDES BLANCOS DEL PERÍTMETRO-------------

    # Definimos el tamaño del margen a recortar.
    final_trim_margin = 70

    # Aplicamos el recorte a la imagen 'morph' (ya erosionada)
    # Aseguramos que el recorte no resulte en una imagen vacía
    if morph.shape[0] > 2 * final_trim_margin and morph.shape[1] > 2 * final_trim_margin:
        morph_trimmed = morph[final_trim_margin:-final_trim_margin, final_trim_margin:-final_trim_margin]
    else:
        print(f"¡Advertencia! El recorte final con margen={final_trim_margin} es demasiado grande. Usando la imagen sin recortar.")
        morph_trimmed = morph.copy() # Si el margen es demasiado grande, usa la imagen original

    plt.figure(figsize=(8, 8))
    plt.imshow(morph_trimmed, cmap="gray")
    plt.title(f"Resultado Final: Erosión + Recorte Exterior ({final_trim_margin}px)")
    plt.axis("off")
    plt.show(block=True)

    #-----------------------SEGMENTAR LOS CARACTERES DEL CROP DE LA PATENTE------------------------------------------

    # 1. Encontrar los componentes conectados y sus estadísticas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        morph_trimmed, connectivity=8
    )

    # Obtenemos las dimensiones de la imagen para el filtrado relativo
    H, W = morph_trimmed.shape
    
    # img_bin_color se usa para la visualización (copia de la binaria en BGR)
    crop_bin_color = cv2.cvtColor(morph_trimmed, cv2.COLOR_GRAY2BGR)
    
    # Lista para almacenar los bounding boxes válidos
    valid_bboxes = [] 

    # 3. Iterar sobre todos los componentes (excepto el fondo, índice 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / float(h)
        
        # ------- FILTROS GEOMÉTRICOS ----------
        
        # Criterio 1: Filtrar por Altura (Altura del carácter vs. Altura de la imagen)
        # Queremos que la altura sea significativa, por ejemplo, entre el 50% y el 100% de la altura total.
        min_char_height = H * 0.40
        max_char_height = H * 1.00 # Máximo igual a la altura total
        
        # Criterio 2: Filtrar por Área
        # Descartar ruido muy pequeño. El área mínima debe ser > 100 píxeles.
        min_area_threshold = 500

        # Area del nounding box
        area_bounding_box = w * h

        # Aplicación de los filtros
        if (h > w and area_bounding_box > min_area_threshold and h >= min_char_height and h <= max_char_height):
            
            # Si cumple los criterios, guardamos el bounding box
            valid_bboxes.append((x, y, w, h))
            
            # Dibujamos el rectángulo rojo
            cv2.rectangle(crop_bin_color, (x, y), (x + w, y + h), (0, 255, 0), 3) # Usamos verde para distinguirlos

    # 7. Mostrar el resultado de los bounding boxes de cada caracter

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(crop_bin_color, cv2.COLOR_BGR2RGB))
    plt.title(f"Componentes Válidos Detectados: {len(valid_bboxes)}")
    plt.axis("off")
    plt.show(block=True)

    #--------------RECORTAR LOS BOUNDING BOXES DEL CROP BINARIZADO SOLO SI HAY 6 ELEMENTOS---------------------

    # NOTA: El recorte se hace sobre el crop binarizado ya que entendemos que se visualizan mejor los caracteres

    if len(valid_bboxes) == 6:
        # 1. Ordenar los bounding boxes de izquierda a derecha (por coordenada 'x')
        valid_bboxes.sort(key=lambda bbox: bbox[0])
        
        # 2. Inicializar la figura para mostrar los 6 caracteres
        fig, axes = plt.subplots(1, 6, figsize=(15, 3))
        fig.suptitle("Segmentación Final de Caracteres", fontsize=16)

        # ---------------------- LE AGREGAMOS UN MAGEN A CADA CROP YA QUE LOS BOUNDING BOXES QUEDAN MUY JUSTOS ----------------------

        # Definir el tamaño del margen a añadir (en píxeles)
        margin_pixels = 120 # Puedes ajustar este valor. Por ejemplo, 5, 10, 15, etc.

        # Obtener las dimensiones de la imagen binarizada reescalada
        # para asegurarnos de que los recortes no se salgan de los límites.
        h_img, w_img = crop_bin.shape[:2] # Asumiendo crop_bin es la imagen binarizada completa

        # 3. Iterar sobre los 6 bounding boxes ya ordenados
        for i, (x, y, w, h) in enumerate(valid_bboxes):
            
            # Aplicar el margen
            x_m = x - margin_pixels
            y_m = y - margin_pixels
            w_m = w + (2 * margin_pixels)
            h_m = h + (2 * margin_pixels)

            # Asegurar que las coordenadas estén dentro de los límites de la imagen (crop_bin)
            x_final = max(0, x_m)
            y_final = max(0, y_m)
            # Asegúrate de que el final del recorte no exceda el ancho/alto de la imagen
            x_final_end = min(w_img, x_m + w_m)
            y_final_end = min(h_img, y_m + h_m)

            # Calcular el nuevo ancho y alto ajustados
            w_final = x_final_end - x_final
            h_final = y_final_end - y_final
            
            # Recortar el carácter de la imagen binarizada
            # Recuerde: la indexación de NumPy es [filas (y):filas+h, columnas (x):columnas+w]
            caracter_crop = crop_bin[y_final : y_final_end, x_final : x_final_end]
            
            # Mostrar el recorte en la sub-gráfica correspondiente
            axes[i].imshow(caracter_crop, cmap='gray') 
            axes[i].set_title(f"Carácter {i+1}")
            axes[i].axis("off")

        plt.show(block=True)
    else:
        # Si no hay 6 bounding boxes encontrados, quiere decir que la función no detecó la patente
        print("NO SE ENCONTRÓ LA PATENTE")

    #--------------------------------------------------------------------------------------------------------------------
    
    # 8. Devolver los bounding boxes válidos
    return valid_bboxes

#------------------------------------------ITERAMOS SOBRE TODAS LAS IMÁGENES--------------------------------
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

imagenes_path = BASE_DIR / "imagenes"
patentes = sorted(p for p in imagenes_path.glob("img*.png"))

#----------------------------------------------------CARGAMOS LA IMAGEN Y LA PASAMOS A LA FUNCIÓN--------------------------------

imagen = cv2.imread(str(patentes[7]))

segmentar_patente(imagen)