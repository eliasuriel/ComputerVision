import cv2
import numpy as np

# Configura la captura de video desde la cámara
cap = cv2.VideoCapture(0)

def mean_shift_segmentation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.pyrMeanShiftFiltering(hsv, 10, 30)
    result = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)
    return result


# Función que aplica el método k-means a una imagen
def kmeans_segmentation(frame):
    # Preprocesamiento de la imagen
    # Configuración de los parámetros del método k-means
    K = 3 # Número de clústeres
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # Criterios de parada
    attempts = 10 # Número de veces que se ejecuta el método con diferentes centroides iniciales
    
    Z = frame.reshape((-1,3))
    Z = np.float32(Z)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convertir los centroides en el tipo de datos correcto y hacer una imagen
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    
    return res2

# Función para procesar el video utilizando el método de Watershed
def water_sheedvideo(frame):
    # Convertir el marco a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar la transformación de umbral
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Aplicar la transformación de apertura
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    # Aplicar la transformación de fondo
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)

    # Aplicar la transformación de detección de área desconocida
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Aplicar la transformación de etiquetado de marcadores
    ret, markers = cv2.connectedComponents(sure_fg)

    # Agregar un marcador para las áreas desconocidas
    markers[unknown==255] = 0

    # Aplicar el método de Watershed
    markers = cv2.watershed(frame,markers)
    frame[markers == -1] = [255,0,0]

    return frame

# Bucle principal para mostrar el video capturado y el video procesado
while(True):
    # Capturar un marco desde la cámara
    ret, frame = cap.read()

    #segmentación mean shift
    ms_frame = mean_shift_segmentation(frame)


    # Mostrar el video capturado
    cv2.imshow('Capturado',frame)

    #mostrar la resultado de la segmentación mean shift
    cv2.imshow('Resultado de la segmentación mean shift', ms_frame)


    #Procesar utilizando el metodo kmeans
    #kmeans_result = kmeans_segmentation(frame)

    #Mostrar resultado del video procesado por kmeans
    #cv2.imshow('K-Means Result', kmeans_result)

    # Procesar el marco utilizando el método de Watershed
    #processed_frame = water_sheedvideo(frame)

    # Mostrar el video procesado
    #cv2.imshow('Watershed',processed_frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
