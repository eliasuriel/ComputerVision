import numpy as np
import cv2 as cv
#Elías Uriel Velázquez Rojas_A01639817
def primeraParte():
    #Parte 1 para cargar las imagenes y hacer la escala de grises 
    images = ['road127.png','road126.png','road125.png','road124.png','road123.png','road122.png','road121.png','road120.png','road119.png','road12.png']
    #Para determinar el filtro aplicado
    Filtro = input("¿Que filtro desea? 1-Avarage 2-Gausiano 3-Median 4-Binomial(ingrese el numero) \n")

    for i in images:
        Icolor = cv.imread(i)
        
        Gesc = cv.cvtColor(Icolor,cv.COLOR_BGR2GRAY) 

        cv.imshow("Imagen a color", Icolor)
        cv.imshow("Imagen a escala de grises", Gesc)

        cv.waitKey(0)
        
        #Parte 2 esta parte se binarizan las imagenes en una escala de grises
        thresh, Binaria = cv.threshold(Gesc,128, 255, cv.THRESH_BINARY)
        cv.imshow("Imagenes binarizadas escala de grises", Binaria)

        cv.waitKey(0)

        #Parte 3 esta parte se binarizan las imagenes dado un canal de color
        _, thresh = cv.threshold(Icolor[:, :, 1],127, 255, cv.THRESH_BINARY)
        
        
        cv.imshow("Imagen Binarizada por umbral de color rojo ", thresh)

        cv.waitKey(0)
        cv.destroyAllWindows()
        #Parte 4 Aqui se aplicaa un ruido gausiano para la escala de grises
        ruido = np.zeros(Gesc.shape, np.int16)
        cv.randn(ruido,0,25)

        #Ruido gausiano
        Gesc_5 = cv.add(Gesc,np.array(ruido *0.05, dtype = np.uint8))
        Gesc_10 = cv.add(Gesc,np.array(ruido *0.1, dtype = np.uint8))
        Gesc_20 = cv.add(Gesc,np.array(ruido *0.2 , dtype = np.uint8))


        cv.imshow("Imagen con 5% de ruido", Gesc_5)
        cv.imshow("Imagen con 10% de ruido", Gesc_10)
        cv.imshow("Imagen con 20% de ruido", Gesc_20)

        cv.waitKey(0)

        #Aplicacion del filtro para la eliminacion del ruido 

        if Filtro == '1':
            #Average filter
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            Fil_img = cv.filter2D(Gesc_5,-1,kernel)
            Fil_img2 = cv.filter2D(Gesc_10,-1,kernel)
            Fil_img3 = cv.filter2D(Gesc_20,-1,kernel)

            cv.imshow("Imagen con filtro  average aplicado ruido 5%", Fil_img)
            cv.imshow("Imagen con filtro  average aplicado ruido 10%", Fil_img2)
            cv.imshow("Imagen con filtro  average aplicado ruido 20%", Fil_img3)
            cv.waitKey(0)
        
        elif Filtro == '2':
            #Filtro Gausiano
            FGaus = cv.GaussianBlur(Gesc_5, (5,5), 0)
            FGaus2 = cv.GaussianBlur(Gesc_10, (5,5), 0)
            FGaus3 = cv.GaussianBlur(Gesc_20, (5,5), 0)
            cv.imshow("Imagen con filtro gausiano aplicado ruido 5%", FGaus)
            cv.imshow("Imagen con filtro gausiano aplicado ruido 10%", FGaus2)
            cv.imshow("Imagen con filtro gausiano aplicado ruido 20$", FGaus3)
            cv.waitKey(0)

        elif Filtro == '3':
            #Filtro mediana
            FMed = cv.medianBlur(Gesc_5, 5)
            FMed2 = cv.medianBlur(Gesc_10, 5)
            FMed3 = cv.medianBlur(Gesc_20, 5)
            cv.imshow("Imagen con filtro median aplicado ruido 5%", FMed)
            cv.imshow("Imagen con filtro median aplicado ruido 10%", FMed2)
            cv.imshow("Imagen con filtro median aplicado ruido 20$", FMed3)
            cv.waitKey(0)
        elif Filtro == '4':
            kernel = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16

            # Aplicar filtro binomial a la imagen
            FBin = cv.filter2D(Gesc_5, -1, kernel)
            FBin2 = cv.filter2D(Gesc_10, -1, kernel)
            FBin3 = cv.filter2D(Gesc_20, -1, kernel)

            cv.imshow("Imagen con filtro binomial en ruido nivel 10", FBin)
            cv.imshow("Imagen con filtro binomial en ruido nivel 20", FBin2)
            cv.imshow("Imagen con filtro binomial en ruido nivel 30", FBin3)
            cv.waitKey(0)
        else:
            print("SYNTAX ERROR")

        cv.destroyAllWindows()

    print("FIN \n" , "Inicia la segunda parte \n")

#primeraParte()

def Segundaparte():

    Edificios = ['ed1.jpg','ed2.jpg','Ed3.jpg','Ed4.jpg','Ed5.jpg']
    Detector = input("Ingrese el sistema de deteccion de bordes 1-Canny 2-Hysteresis \n ")

    for i in range(1):
        Ecolor = cv.imread('Ed1.jpg')
        Egray = cv.cvtColor(Ecolor,cv.COLOR_BGR2GRAY)

        cv.imshow('Imagen de edificio a color', Ecolor)

        if Detector == '1':
            canny = cv.GaussianBlur(Egray, (3, 3), 0)
            ejes = cv.Canny(canny, 100, 200, apertureSize=5)
            cv.imshow('Detector de bordes por canny', ejes)
            cv.waitKey(0)

            #Operaciones morfologicas
            Kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
            #Erocion
            c = cv.morphologyEx(ejes, cv.MORPH_CLOSE,Kernel)
            ero = cv.erode(c,Kernel,iterations=1)
            cv.imshow('Operacion de erosion',ero)
            cv.waitKey(0)
            #Dilatacion
            Dilate = cv.dilate(ejes, Kernel, iterations=1)
            cv.imshow('Operacion de dilatacion',Dilate)   
            cv.waitKey(0)
            #Apertura
            cerr = cv.morphologyEx(ejes,cv.MORPH_CLOSE,Kernel)
            open = cv.morphologyEx(cerr,cv.MORPH_OPEN,Kernel)
            cv.imshow('Operacion de apertura',open)    
            cv.waitKey(0)
            #Cierre
            close = cv.morphologyEx(ejes, cv.MORPH_CLOSE, Kernel)
            cv.imshow('Operacion de cierre',close)
            cv.waitKey(0)
            cv.destroyAllWindows()   

        elif Detector == '2':
            His = cv.GaussianBlur(Egray, (3,3), 0)
            sobelx = cv.Sobel(His, cv.CV_64F, 1, 0, ksize=3)
            sobely = cv.Sobel(His, cv.CV_64F, 0, 1, ksize=3)  
            mag, ang = cv.cartToPolar(sobelx, sobely, angleInDegrees=True)  
            low_lim = 50
            up_lim= 150
            ejesH = np.zeros_like(Ecolor)
            ejesH[(mag >= low_lim) & (mag <= up_lim)] = 255
            ejesH = cv.dilate(ejesH, None)
            ejesH = cv.erode(ejesH, None)
            
            cv.imshow('Detector de bordes por Hysteresis Thresholding',ejesH)
            cv.waitKey(0)

            #Operaciones morfologicas
            Kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            #Erosion
            Erode = cv.erode(ejesH,Kernel, iterations=1)
            cv.imshow('Operacion de erosion',Erode)
            cv.waitKey(0)
            #Dilatacion
            Dilate = cv.dilate(ejesH,Kernel, iterations=1)
            cv.imshow('Operacion de dilatacion',Dilate)  
            cv.waitKey(0) 
            #Apertura
            open = cv.morphologyEx(ejesH, cv.MORPH_OPEN, Kernel)
            cv.imshow('Operacion de apertura',open)    
            cv.waitKey(0)
            #cierre
            close = cv.morphologyEx(ejesH, cv.MORPH_CLOSE, Kernel)
            cv.imshow('Operacion de cierre',close)         
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("SYNTAX ERROR")

#Segundaparte()



cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #Pasar la imagen a escala de grises 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Parametros para la detección de bordes
    edges = cv.Canny(gray, 50, 150, apertureSize=3) #Detecta los bordes en la imagen en escala de grises utilizando el algoritmo Canny
    edges_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR) #Convierte la imagen de bordes a una imagen en color.
    edges_color[np.where((edges_color == [255, 255, 255]).all(axis=2))] = [0, 255, 255] #Resalta los bordes detectados.

    #Mostrar las imágenes en la pantalla
    cv.imshow('Escala de Grises', gray)
    cv.imshow('Bordes resaltados', cv.addWeighted(frame, 0.8, edges_color, 0.2, 0))

    #Detectar círculos en la imagen
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=0, maxRadius=0)
    #Detectar lineas en la imagen
    lines = cv.HoughLines(edges, 1, theta=np.pi/180, threshold=100)

    #Mostrar los circulos en frame
    if circles is not None: #Condicional para que no regrese una matriz vacia si no encuentra circulos
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            #Resalta los circulos encontrados en verde
            cv.circle(frame, (x, y), r, (0, 255, 0), 2)

    #Mostrar los circulos en frame
    if lines is not None: #Condicional para que no regrese una matriz vacia si no encuentra lineas
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            #Resalta las lineas en rojo
            cv.line(frame, (x1,y1), (x2,y2), (0,0,255), 2)
    
    #Binarización de los bordes para una detección mas facil
    ret, thrash = cv.threshold(edges, 240 , 255, cv.CHAIN_APPROX_NONE)
    #Encuentra los bordes con la funcion findContours
    contours , hierarchy = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  

    #Resalta los bordes encontrados en azul
    cv.drawContours(frame, contours, -1, (255, 0, 0), 3)

    #Muestra el video con los poligonos resaltados en la pantalla
    cv.imshow('Circulos, lineas y poligonos', frame)
    
    #Presionar q para salir
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()