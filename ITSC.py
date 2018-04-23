import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pyaudio
import wave

def mostrarImagenInicialEstandard(nombre, x,y):
    imagen = cv2.imread(nombre)
    imagenStandard = cv2.resize(imagen, (x,y))
    cv2.imshow("Imagen a Convertir",imagenStandard)  
    return imagenStandard

def obtenerMatricesBGR(imagenStandard, x,y):
    b = np.zeros((y,x))
    g = np.zeros((y,x))
    r = np.zeros((y,x))

    for n in list(range(y)):
        for m in list(range(x)):
            b[n][m] = ((imagenStandard[n][m])[0]) 
            g[n][m] = ((imagenStandard[n][m])[1]) 
            r[n][m] = ((imagenStandard[n][m])[2])         

    cv2.imwrite("recursosImg/rgb/blue.jpg", b)
    cv2.imwrite("recursosImg/rgb/green.jpg", g)
    cv2.imwrite("recursosImg/rgb/red.jpg", r)

    bgr = [b,g,r]
    return bgr

def obtenerPixelLbp(color, n, m):
    exponente = 0
    exponentes = [6,7,0,1,2,3,4,5]
    pixelLpbB = 0
    pixelLpbG = 0
    pixelLpbR = 0
    valorCentral0 = (color[0])[n][m]
    valorCentral1 = (color[1])[n][m]
    valorCentral2 = (color[2])[n][m]

    for k in list(range(n-1,n+2)):
        for j in list(range(m-1,m+2)):
            if k != n and j != m:
                if (color[0])[k][j] <= valorCentral0:
                    pixelLpbB = pixelLpbB + pow(2,exponentes[exponente])
                if (color[1])[k][j] <= valorCentral1:
                    pixelLpbG = pixelLpbG + pow(2,exponentes[exponente])
                if (color[2])[k][j] <= valorCentral2:
                    pixelLpbR = pixelLpbR + pow(2,exponentes[exponente])
                exponente = exponente+1

    return [pixelLpbB,pixelLpbB,pixelLpbR]            

def obtenerColor(imagenStandard, n , m):
    color =[ 0,0,0]
    for k in list(range(n-1,n+2)):
        for j in list(range(m-1,m+2)):
            color[0] = color[0]+(imagenStandard[n][m])[0] 
            color[1] = color[1]+(imagenStandard[n][m])[1] 
            color[2] = color[2]+(imagenStandard[n][m])[2]

    color = [int(color[0]/9),int(color[1]/9),int(color[2]/9)]

    return color

def obtenerValoresConversion(imagenStandard,bgr,x,y, compresionNumber):
    cn = 0
    matOrigen= bgr
    valoresConversion=[]

    while cn <= compresionNumber:
        puntosX = int((x-1)/3)
        puntosY = int((y-1)/3)

        lbpB = np.zeros((puntosY ,puntosX))
        lbpG = np.zeros((puntosY ,puntosX))
        lbpR = np.zeros((puntosY ,puntosX))
        #colores = np.ndarray((puntosY ,puntosX))
        colores = x = [[ [0,0,0] for i in range(puntosX)] for j in range(puntosY)]

        if cn == compresionNumber:
            lbpU = np.zeros((puntosY ,puntosX))
            lbpF = np.zeros((puntosY ,puntosX)) 
            canal = np.zeros((puntosY ,puntosX))          

            view = np.zeros((puntosY ,puntosX))

            sonidoPorPixelI = np.zeros((puntosY ,puntosX))
            sonidoPorPixelF = np.zeros((puntosY ,puntosX))
            sonidoPorPixelM = np.zeros((puntosY ,puntosX))

        for n in list(range(1,puntosY +1)):
            for m in list(range(1,puntosX +1)):
                o = 0
                p = 0
                if m != 1:
                    o = 3
                if n != 1:
                    p = 3
                lbpS = obtenerPixelLbp(matOrigen, n+p, m+o) 
                lbpB[n-1][m-1] = lbpS[0]
                lbpG[n-1][m-1] = lbpS[1]
                lbpR[n-1][m-1] = lbpS[2]

                colores[n-1][m-1]= obtenerColor(imagenStandard,n+p,m+o)
                
                if cn == compresionNumber:
                    d = colores[n-1][m-1]#revisar

                    lbpU[n-1][m-1] = lbpB[n-1][m-1] +lbpG[n-1][m-1] +lbpR[n-1][m-1] 
                    lbpF[n-1][m-1] = lbpU[n-1][m-1] + d[0] + d[1]+ d[2]
                    view[n-1][m-1] = (lbpF[n-1][m-1]) *0.166
                                      

                    sonidoPorPixelI[n-1][m-1] = 40+9*lbpF[n-1][m-1]+lbpF[n-1][m-1]
                    sonidoPorPixelF[n-1][m-1] = sonidoPorPixelI[n-1][m-1] +9
                    sonidoPorPixelM[n-1][m-1] = sonidoPorPixelI[n-1][m-1] +4

                    print("testo")
                    print(d[0])
                    print(d[1])
                    print(d[2])

                    if d[0]> d[1] and d[0] > d[2]:
                        canal[n-1][m-1] = 0
                    elif d[2] > d[1] and d[2] > d[0]:
                        canal[n-1][m-1] = 2
                    else:
                        canal[n-1][m-1] = 1

                    print("canal")
                    print(canal[n-1][m-1])
                    print(canal)

                    valoresConversion = [lbpB,lbpG,lbpR, lbpU,lbpF, canal, sonidoPorPixelI, sonidoPorPixelF, sonidoPorPixelM, puntosX,puntosY]

        matOrigen = [lbpB,lbpG,lbpR]
        x = puntosX
        y = puntosY
        imagenStandard = colores
        cn = cn+1    

    print(lbpF)
    print("el toro")
    print(lbpU)

    cv2.imwrite("recursosImg/lpbs/lbpBC.jpg", lbpB)
    cv2.imwrite("recursosImg/lpbs/lbpGC.jpg", lbpG)
    cv2.imwrite("recursosImg/lpbs/lbpRC.jpg", lbpR)
    cv2.imwrite("recursosImg/lpbs/lbpUC.jpg", lbpU)
    cv2.imwrite("recursosImg/lpbs/lbpFC.jpg", lbpF)
    cv2.imwrite("recursosImg/viewBWC.jpg", view)

    return valoresConversion

def onda(frecuencia, duracion, rate=44100):
    duracion = int(duracion * rate)
    factor = float(frecuencia) * (math.pi * 2) / rate
    return np.sin(np.arange(duracion) * factor)

def reproducir(stream, senial):
    partes = []
    partes.append(senial)

    parte =np.concatenate(partes) * 0.25
    stream.write(parte.astype(np.float32).tostring())

#if __name__ == '__main__':

def obtenerSonidoDeImagen(valoresConversion, numSeg):
    sonidoF = []
    print(valoresConversion[9])
    print(valoresConversion[10])
    sonidoPorPixelM = valoresConversion[8]
    canal = valoresConversion[5]
    print(canal)
    cv2.waitKey(0)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,channels=1, rate=44100, output=1)
    for n in list(range(valoresConversion[10])):
        for m in list(range(valoresConversion[9])):            
            print(n)
            print(m)
            print(sonidoPorPixelM[n][m])
            print(canal[n][m])

            senial = onda(sonidoPorPixelM[n][m],numSeg/(valoresConversion[9]*valoresConversion[10]))
            senial2 = onda(40,numSeg/(valoresConversion[9]*valoresConversion[10]))

            if canal[n][m] == 0 :
                senial_stereo = np.ravel(np.column_stack((senial,senial2)))
            elif canal[n][m] == 1:
                senial_stereo = np.ravel(np.column_stack((senial,senial)))
            else:
                senial_stereo = np.ravel(np.column_stack((senial2,senial)))

            reproducir(stream,senial_stereo )

    stream.close()
    p.terminate()

    return sonidoF

def inicio(nombreImagen, numSeg, x, y,compresionNumber):
    img = mostrarImagenInicialEstandard(nombreImagen, x,y)
    bgr = obtenerMatricesBGR(img, x,y)
    valoresConversion= obtenerValoresConversion(img,bgr,x,y,compresionNumber)

    sonidoDeImagen = obtenerSonidoDeImagen(valoresConversion, numSeg)
    cv2.waitKey(0)

#inicio("srcImagenes/carito.jpg", 60, 200,150, 3)
inicio("srcImagenes/escalaX.jpg", 15, 400,300, 2)

print("Graciassss TOTALES!!")
