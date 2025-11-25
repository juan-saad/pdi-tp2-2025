from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path.cwd()
main()
imagenes_path = BASE_DIR / "imagenes"
patentes = sorted(p for p in imagenes_path.glob("img*.png"))

# --- INICIO DE PROCESAMIENTO ---

#-----------------------------------------------------------

# 1. Cargar la primera imagen para el ejemplo inicial
imagen_a_procesar_path = patentes[10]
print(f"Cargando imagen: {imagen_a_procesar_path.name}")

# Leer la imagen usando OpenCV (en formato BGR por defecto)
imagen_color = cv2.imread(str(imagen_a_procesar_path))

# Mostrar la imagen cargada
#cv2.imshow("1. Imagen Original (Color)", imagen_color)

# 2. Mostrar la imagen usando Matplotlib
plt.figure(figsize=(10, 6)) # Opcional: define el tamaño de la figura
plt.imshow(imagen_color)
plt.title(f"Imagen Original (Matplotlib) - {imagen_color}")
plt.axis('off') # Opcional: oculta los ejes para una mejor visualización de la imagen
plt.show() # Muestra la ventana con la imagen

#------------------------------------------------------------

# 2. Convertir la imagen a escala de grises
# cv2.COLOR_BGR2GRAY es la conversión estándar para imágenes cargadas con cv2.imread
imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

# Mostrar la imágen en escala de grises
#cv2.imshow("2. Imagen en Escala de Grises", imagen_gris)

#-------------------------------------------------------------

# 3. Aplicar Filtro Gaussiano para Reducción de Ruido
# Kernel (5, 5) es un buen punto de partida para suavizar
imagen_suavizada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
#cv2.imshow("3. Imagen con filtro Gaussiano", imagen_suavizada)

#-------------------------------------------------------------

# 4- Binarizar la imagen con Binarización Adaptativa 
imagen_binaria = cv2.adaptiveThreshold(
    imagen_suavizada, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 
    11, 
    2
)

#cv2.imshow("4. Imagen Binarizada", imagen_binaria)

#-------------------------------------------------------------

# 5- Detección de contronos
contornos, jerarquia = cv2.findContours(
    imagen_binaria.copy(), 
    cv2.RETR_LIST, 
    cv2.CHAIN_APPROX_SIMPLE
)

# Crear una copia de la imagen a color para dibujar los contornos encontrados.
imagen_contornos = imagen_color.copy()

# Dibujar todos los contornos encontrados (propósito de depuración/visualización)
imagen_contonoeada =cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 1)

#cv2.imshow("5. Contornos de la Imagen", imagen_contornos)

# 2. Mostrar la imagen usando Matplotlib
plt.figure(figsize=(10, 6)) # Opcional: define el tamaño de la figura
plt.imshow(imagen_contonoeada)
plt.title(f"Imagen Original (Matplotlib) - {imagen_contonoeada}")
plt.axis('off') # Opcional: oculta los ejes para una mejor visualización de la imagen
plt.show() # Muestra la ventana con la imagen

imagen_canny = cv2.Canny(imagen_gris, 150, 500)

for p in patentes:
    canny = cv2.Canny(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), 150, 500)
    plt.figure(figsize=(10, 6)) # Opcional: define el tamaño de la figura
    plt.imshow(canny, cmap='gray')
    plt.title(f"Imagen Original (Matplotlib) - {p.name}")
    plt.axis('off') # Opcional: oculta los ejes para una mejor visual
    plt.show()



# # 6- Filtrado de contornos con mayor robustez
# # Filtros de área Máxima y Mínima (se mantienen o se ajustan si es necesario)
# MIN_AREA = 100  # Área mínima para descartar ruido
# MAX_AREA = 30000 # Ligeramente más amplio que antes

# # --- FILTROS DE RELACIÓN DE ASPECTO INICIAL (Más amplio para robustez) ---
# # Aceptamos contornos que están 'cerca' de ser rectángulos
# ASPECT_RATIO_MIN = 1.5 
# ASPECT_RATIO_MAX = 3.0 


# contornos_candidatos = []

# for c in contornos:
#     # 1. Encontrar el rectángulo delimitador para w y h (no necesitamos approxPolyDP para esto)
#     x, y, w, h = cv2.boundingRect(c)
    
#     # 2. Cálculo de Propiedades
#     area = cv2.contourArea(c)
#     aspect_ratio = w / h
    
#     # 3. Cálculo de la Solidez
#     # Obtener el Convex Hull
#     hull = cv2.convexHull(c)
#     hull_area = cv2.contourArea(hull)
    
#     # Prevenir división por cero
#     if hull_area == 0:
#         solidity = 0
#     else:
#         solidity = area / hull_area 
    
#     # 4. Aplicar los Filtros Heurísticos
#     if (area > MIN_AREA and 
#         area < MAX_AREA and
#         aspect_ratio >= ASPECT_RATIO_MIN and 
#         aspect_ratio <= ASPECT_RATIO_MAX and
#         solidity >= MIN_SOLIDITY):  # <--- NUEVO FILTRO CLAVE
        
#         # Si el contorno cumple con todos los criterios
#         contornos_candidatos.append((x, y, w, h, c, solidity))



# --- 1. DEFINICIÓN DE CONSTANTES ---
MIN_AREA = 100
MAX_AREA = 30000 
ASPECT_RATIO_MIN = 1.5 
ASPECT_RATIO_MAX = 5.0 # Aumenté un poco el máximo por si la patente es muy ancha

# Suponiendo que 'patentes' es tu lista de rutas a las imágenes
# for p in patentes: 
# (Uso una lista ficticia para el ejemplo, descomenta tu bucle arriba)

for p in patentes:
    # A. Leer imagen y preprocesar
    img_bgr = cv2.imread(str(p))
    if img_bgr is None: continue # Seguridad por si la ruta falla
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # B. Aplicar Canny (Tus parámetros)
    # Nota: 500 es muy alto para el umbral superior (el max suele ser 255), 
    # pero si te funciona para tu iluminación, adelante.
    imagen_canny = cv2.Canny(img_gray, 150, 500) 

    # --- PASO FALTANTE CRUCIAL: ENCONTRAR CONTORNOS ---
    # Buscamos contornos EXTERNOS en la imagen de bordes (Canny)
    contornos, _ = cv2.findContours(imagen_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos_candidatos = []

    # C. Filtrado de Contornos
    for c in contornos:
        # 1. Propiedades geométricas
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Evitar divisiones por cero
        if h == 0: continue
        aspect_ratio = w / float(h)

              
        # 3. Filtros Heurísticos
        if (area > MIN_AREA and area < MAX_AREA and
            aspect_ratio >= ASPECT_RATIO_MIN and aspect_ratio <= ASPECT_RATIO_MAX):
            
            # Si pasa el filtro, lo guardamos
            contornos_candidatos.append(c)

    # --- VISUALIZACIÓN DE RESULTADOS ---
    
    # Copia para dibujar sin dañar la original
    img_resultado = img_bgr.copy()
    
    # Dibujamos los contornos que pasaron el filtro en VERDE (grosor 2)
    cv2.drawContours(img_resultado, contornos_candidatos, -1, (0, 255, 0), 2)

    # Plotting con Matplotlib
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Canny
    plt.subplot(1, 2, 1)
    plt.imshow(imagen_canny, cmap='gray')
    plt.title("Bordes Detectados (Canny)")
    plt.axis('off')

    # Subplot 2: Resultado Filtrado
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
    plt.title(f"Candidatos Filtrados: {len(contornos_candidatos)}")
    plt.axis('off')
    
    plt.show()