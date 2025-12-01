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


    # Cargamos la imagen y la convertimos a escala de grises
    img = cv2.imread(str(IMAGE_PATH))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Top-hat
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    img_tophat = cv2.morphologyEx(img_gray, kernel=kernel_tophat, op=cv2.MORPH_TOPHAT)

    # Gradiente morphológico
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    img_grad = cv2.morphologyEx(img_tophat, cv2.MORPH_GRADIENT, kernel_grad)

    # Binarización Otsu
    _ , img_bin = cv2.threshold(img_grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_bin = cv2.GaussianBlur(img_bin, (5, 5), 2)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(img_bin, cv2.COLOR_BGR2RGB))
    plt.title("Imagen binarizada luego de Top-hat y Gradiente morphológico")
    plt.axis("off")
    plt.show()

    MIN_AREA = 3000     # umbral de área mínimo

    # contornos
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # filtramos contornos por área
    contours_filtrados = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

    # imagen mismo tamaño que la original, para dibujar contornos
    mask_contornos = np.zeros_like(img_bin)

    # dibujo de contornos filtrados
    cv2.drawContours(mask_contornos, contours_filtrados, -1, 255, -1)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(mask_contornos, cv2.COLOR_BGR2RGB))
    plt.title("Contornos detectados y filtrados")
    plt.axis("off")
    plt.show()

    # aplico canny para diferenciar bordes
    contornos_canny = cv2.Canny(mask_contornos, 50, 150)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(contornos_canny, cv2.COLOR_BGR2RGB))
    plt.title("canny sobre contornos filtrados")
    plt.axis("off")
    plt.show()

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


    # Buscamos objetos conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_hough, connectivity=8)

    mask_filtrada = np.zeros_like(img_hough)

    # Limpiamos objetos pequeños
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= MIN_AREA:
            mask_filtrada[labels == label] = 255

    # Volvemos a buscar sobre la máscara filtrada
    _ , _ , stats_final, _ = cv2.connectedComponentsWithStats(mask_filtrada, connectivity=8)

    # Definimos listas para cada tipo de moneda y guardamos las areas salvo la del fondo
    areas = stats_final[1:, cv2.CC_STAT_AREA] 
    moneda_10 = []
    moneda_50 = []
    moneda_1 = []

    # Definimos umbrales de área para clasificar las monedas
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



    # Empezamos a trabajar la parte de dados, cambiando por a negro los contornos de las monedas
    AmB = img_gray.copy()
    AmB[mask_filtrada>0]=0
    plt.imshow(AmB, cmap='gray')
    plt.title("Contornos detectados")
    plt.axis("off")
    plt.show()

    # Aplicamos canny
    dados_canny = cv2.Canny(AmB, 50, 150)
    plt.imshow(dados_canny, cmap='gray')
    plt.title("Contornos detectados")
    plt.axis("off")
    plt.show()


    # Buscamos objetos conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dados_canny, connectivity=8)
    img_etiquetas = cv2.cvtColor(dados_canny, cv2.COLOR_GRAY2BGR)

    # Fijamos umbrales de área para filtrar objetos no deseados
    MIN_AREA_DADOS = 180 
    MAX_AREA_DADOS = 250
    mask_dados_filtrados = np.zeros_like(dados_canny)

    # Filtamos objetos por área para eliminar ruido
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= MIN_AREA_DADOS and area <= MAX_AREA_DADOS:
            mask_dados_filtrados[labels == label] = 255
        

    plt.figure(figsize=(6,6))
    plt.imshow(mask_dados_filtrados, cmap='gray')
    plt.title("Dados filtrados por área mínima")
    plt.axis("off")
    plt.show()


    # Usamos Hough para detectar los círculos de los pips
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

    # Filtramos ruido por area mínima para aislar los pips
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

    # Calculamos la cantidad de dados segundo la distancia entre los centroides de los objetos
    num_labels_pips_dados , labels_pips_dados , stats_pips_dados, centroids_pips_dados = cv2.connectedComponentsWithStats(mask_pips_filtrados, connectivity=8)

    dic_pips = {}
    dic_pips = {"dado_1":[(int(centroids_pips_dados[1][0]),int(centroids_pips_dados[1][1]))]} 


    for label in range(2, num_labels_pips_dados):
        cx, cy = int(centroids_pips_dados[label][0]), int(centroids_pips_dados[label][1])
        distancia_menor = float('inf')
        dado_asignado = None

        for dado, pip in dic_pips.items():
            for pip_label in pip:           
                bx, by = int(pip_label[0]), int(pip_label[1])
                dx = cx - bx
                dy = cy - by
                distancia = np.sqrt(dx*dx + dy*dy) # Usamos distancia euclidiana

                if distancia < distancia_menor:
                    distancia_menor = distancia
                    dado_asignado = dado

        if distancia_menor < 200 :  # Umbralamos la distancia para asignar pips al mismo dado
            dic_pips[dado_asignado].append((cx,cy))
        else:
            dic_pips[f"dado_{len(dic_pips)+1}"] = [(cx,cy)]

    print("Cantidad de dados detectados y sus pips:")
    for dado, pips in dic_pips.items():
        print(f"{dado}: {len(pips)} pips")


    # Empezamos a trabajar los bounding box
    bounding_box_monedas = mask_filtrada
    bounding_box_monedas_canny = cv2.Canny(bounding_box_monedas, 50, 150)

    # Engrosamos los bordes para mejor visualización
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Ajusta el tamaño para más grosor
    bounding_box_monedas_canny = cv2.dilate(bounding_box_monedas_canny, kernel_dilatacion, iterations=1)

    # Superponemos los bounding box en la imagen original
    img_superpuesta = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_superpuesta[bounding_box_monedas_canny > 0] = [0, 0, 255]

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(img_superpuesta, cv2.COLOR_BGR2RGB))
    plt.title("Bounding box monedas (Canny grueso) superpuesto en verde")
    plt.axis("off")
    plt.show()


    centros = {}
    # Calculamos el pips central de cada dado para dibujar el bounding box alrededor
    for nombre_dado, puntos in dic_pips.items():
        puntos_np = np.array(puntos)
        centro = np.mean(puntos_np, axis=0)
        centros[nombre_dado] = tuple(centro.astype(int))

    for nombre_dado, centro in centros.items():
        label_centro = labels_pips_dados[centro[1], centro[0]]
        if label_centro == 0:
            continue

        mask_objeto = np.zeros_like(mask_pips_filtrados)
        mask_objeto[labels_pips_dados == label_centro] = 255

        # Agrandamos el boundingbox del pip para abarcar todo el dado
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (400, 400))
        mask_objeto_grande = cv2.dilate(mask_objeto, kernel, iterations=1)

        # Dibujamos el contorno del bounding box del dado en la imagen superpuesta
        contornos, _ = cv2.findContours(mask_objeto_grande, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_superpuesta, contornos, -1, (0,255,0), 2)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(img_superpuesta, cv2.COLOR_BGR2RGB))
    plt.title("Bounding box monedas y contornos dados superpuestos")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()










