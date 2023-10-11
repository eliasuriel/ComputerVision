from skimage.feature import blob_dog
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np

# Leer la imagen
image = imread('ruta/de/tu/imagen.jpg')

# Convertir la imagen a escala de grises
image_gray = rgb2gray(image)

# Aplicar el método blobs para detectar características
blobs = blob_dog(image_gray, max_sigma=30, threshold=.1)

# Crear una matriz de datos para entrenar el modelo de clasificación
features = np.zeros((len(blobs), 3))
for i, blob in enumerate(blobs):
    y, x, r = blob
    features[i, 0] = y
    features[i, 1] = x
    features[i, 2] = r

# Entrenar el modelo de clasificación
# Aquí deberías elegir el modelo de clasificación que mejor se adapte a tus datos y necesidades
# y utilizar la matriz de datos "features" como entrada al modelo de entrenamiento