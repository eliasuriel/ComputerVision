import numpy as np
import cv2 
import os

images = ['road0.png','road1.png','road10.png','road100.png','road101.png','road102.png','road103.png','road104.png','road105.png','road106.png']

#Metodos de segmentacion
def metodo_watersheed(images):   
    #Parte 1 para cargar las imagenes y hacer la escala de grises 
 
    for i in images:
        img1 = cv2.imread(i)
        img = cv2.convertScaleAbs(img1)
        # Convertimos la imagen a escala de grises
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Aplicamos un filtro de suavizado para eliminar ruido
        blur = cv2.medianBlur(gray,5)

        # Binarizamos la imagen para obtener una imagen en blanco y negro
        ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Aplicamos la transformación morfológica de cierre para unir regiones
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations = 2)

        # Encontramos los contornos de la imagen
        contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Creamos una matriz de marcadores y los inicializamos con ceros
        markers = np.zeros((img.shape[0], img.shape[1]),dtype=np.int32)

        # Dibujamos los contornos encontrados en los marcadores
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i+1), -1)

        # Agregamos un valor constante a los marcadores para separar los objetos
        markers = markers + 1

        # Aplicamos el método Watershed para segmentar la imagen
        markers = cv2.watershed(img, markers)

        # Coloreamos los marcadores para visualizar las regiones segmentadas
        img[markers == -1] = [255,0,0]

        # Mostramos la imagen original y la segmentada
        cv2.imshow("Original",img)
        # Convertimos la matriz de marcadores a una imagen de escala de grises
        markers_img = cv2.convertScaleAbs(markers)

        # Aplicamos una escala de colores a la imagen de marcadores
        colormap = cv2.applyColorMap(markers_img, cv2.COLORMAP_JET)

        # Mostramos la imagen segmentada
        cv2.imshow("Segmentada", colormap) 

        # Esperamos a que el usuario presione una tecla para cerrar las ventanas
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def K_means(images): 
    for i in images:
        img = cv2.imread(i)
        img_float = np.float32(img) / 255.0

        # Definimos los parámetros del método k-means
        num_clusters = 4
        max_iteraciones = 10
        epsilon = 1.0

        # Aplicamos el método k-means
        resultado = cv2.kmeans(img_float.reshape(-1, 3), num_clusters, None, 
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iteraciones, epsilon), 
                            attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

        # Obtenemos las etiquetas de cada pixel y los centroides de cada cluster
        labels = resultado[1]
        centroids = resultado[2]

        # Asignamos a cada pixel el valor del centroide de su cluster correspondiente
        segmentado = centroids[labels.flatten()].reshape(img.shape)

        # Convertimos la imagen segmentada a valores enteros entre 0 y 255
        segmentado = np.uint8(segmentado * 255)

        # Mostramos la imagen original y la imagen segmentada
        cv2.imshow("Original", img)
        cv2.imshow("Segmentada", segmentado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def Mean_shift(images):
    for i in images:
        img = cv2.imread(i)

        # Redimensionamos la imagen para acelerar el proceso de segmentación
        img2 = cv2.pyrMeanShiftFiltering(img, 20, 45)

        # Convertimos la imagen a escala de grises
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Aplicamos una binarización a la imagen
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Encontramos los contornos en la imagen binarizada
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Dibujamos los contornos en la imagen original
        for cnt in contours:
            cv2.drawContours(img2, [cnt], 0, (0, 0, 255), 2)

        # Mostramos la imagen segmentada
        cv2.imshow("Original",img)
        cv2.imshow("Segmentada", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#ruido Gausiano mas todos los metodos
def ruido_Gaus(Iruido):
    seg = input('Ingrese el metodo de segmentacion que desee 1-k-means 2-Mean-shift 3-watersheed: ')
    for i in Iruido:
        img = cv2.imread(i)
        img1 = cv2.convertScaleAbs(img)
        # Convertimos la imagen a escala de grises
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        ruido = np.zeros(gray.shape, np.int16)
        #ruido Gausiano
        cv2.randn(ruido,0,25)

        #Ruido gausiano
        Gesc_5 = cv2.add(gray,np.array(ruido *0.05, dtype = np.uint8))
        Gesc_10 = cv2.add(gray,np.array(ruido *0.1, dtype = np.uint8))
        Gesc_20 = cv2.add(gray,np.array(ruido *0.2 , dtype = np.uint8))


        # Creamos una lista de nombres de archivo válidos
        filenames = [f"{i}_{j}.png" for j in range(3)]
        # Guardamos las imágenes con nombres de archivo válidos
        cv2.imwrite(filenames[0], Gesc_5)
        cv2.imwrite(filenames[1], Gesc_10)
        cv2.imwrite(filenames[2], Gesc_20)

        # Llamamos a metodo_watersheed con los nombres de archivo válidos
        if seg == '1':
            K_means(filenames)
        elif seg == '2':
            Mean_shift(filenames)
        elif seg == '3':
            metodo_watersheed(filenames)
        else:
            print('metodo no encontrado')
            break     

def video():
    seg = input('Ingrese el metodo de segmentacion por el que desea imprimir: 1-k-means 2-mean_shift 3-watersheed: ')
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
        kernel = np.ones((1,1), np.uint8)
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
        if seg == '1':
            ms_frame = kmeans_segmentation(frame)
        elif seg == '2':
            ms_frame = mean_shift_segmentation(frame)
        elif seg == '3':
            ms_frame = water_sheedvideo(frame)

        # Mostrar el video capturado
        cv2.imshow('Capturado',frame)

        #mostrar la resultado de la segmentación mean shift
        #cv2.imshow('Resultado de la segmentación mean shift', ms_frame)


        #Procesar utilizando el metodo kmeans
        if seg == '1':
            kmeans_result = kmeans_segmentation(frame)
            cv2.imshow('K-Means Result', kmeans_result)
        elif seg == '2':
            kmeans_result = mean_shift_segmentation(frame)
            cv2.imshow('mean_shift', kmeans_result)
        elif seg == '3':
            kmeans_result = water_sheedvideo(frame)
            cv2.imshow('water_sheed', kmeans_result)
        

        #Mostrar resultado del video procesado por kmeans

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

K_means(images)
Mean_shift(images)
metodo_watersheed(images)
ruido_Gaus(images)
video()