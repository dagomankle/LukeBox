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
    #plt.imshow(imagenStandard, cmap= "gray", interpolation="bicubic")
    #plt.show()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return imagenStandard

def obtenerMatricesBGR(imagenStandard, x,y):
    b = np.zeros((y,x))
    g = np.zeros((y,x))
    r = np.zeros((y,x))
    #bp = np.array((300,400),np.dtype(np.zeros((3)))#np.dtype({'col1': (int,0),'col2': (int,6),'col3': (int,12)}) )#np.zeros((300,400))
    #gp = np.zeros((300,400))
    #rp = np.zeros((300,400))

    for n in list(range(y)):
        for m in list(range(x)):
            b[n][m] = ((imagenStandard[n][m])[0]) 
            g[n][m] = ((imagenStandard[n][m])[1]) 
            r[n][m] = ((imagenStandard[n][m])[2]) 
           # a = ((imagenStandard[n][m])[0])
            #bp[n][m] = np.array([a, 0, 0 ])
            #gp[n][m] = np.array([0,((imagenStandard[n][m])[1]),0])
            #rp[n][m] = np.array([0,0,((imagenStandard[n][m])[2])])           

    cv2.imwrite("recursosImg/rgb/blue.jpg", b)
    cv2.imwrite("recursosImg/rgb/green.jpg", g)
    cv2.imwrite("recursosImg/rgb/red.jpg", r)

    bgr = [b,g,r]
    #print((imagenStandard[0][0])[0])
    #print(bgr[0])
    #print(imagenStandard)

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

def obtenerValoresConversion(imagenStandard,bgr,x,y):
    lbpB = np.zeros((y-2,x-2))
    lbpG = np.zeros((y-2,x-2))
    lbpR = np.zeros((y-2,x-2))
    lbpU = np.zeros((y-2,x-2))
    lbpF = np.zeros((y-2,x-2))
    canal = np.zeros((y-2,x-2))

    view = np.zeros((y-2,x-2))

    sonidoPorPixelI = np.zeros((y-2,x-2))
    sonidoPorPixelF = np.zeros((y-2,x-2))
    sonidoPorPixelM = np.zeros((y-2,x-2))

    for n in list(range(1,y-1)):
        for m in list(range(1,x-1)):
            d = imagenStandard[n][m]

            lbpS = obtenerPixelLbp(bgr, n, m) 
            lbpB[n-1][m-1] = lbpS[0]
            lbpG[n-1][m-1] = lbpS[1]
            lbpR[n-1][m-1] = lbpS[2]

            lbpU[n-1][m-1] = lbpB[n-1][m-1] +lbpG[n-1][m-1] +lbpR[n-1][m-1] 
            lbpF[n-1][m-1] = lbpU[n-1][m-1] + d[0] + d[1]+ d[2]
            view[n-1][m-1] = (lbpF[n-1][m-1]) *0.166

            sonidoPorPixelI[n-1][m-1] = 40+9*lbpF[n-1][m-1]+lbpF[n-1][m-1]
            sonidoPorPixelF[n-1][m-1] = sonidoPorPixelI[n-1][m-1] +9
            sonidoPorPixelM[n-1][m-1] = sonidoPorPixelI[n-1][m-1] +4

            if d[0]> d[1] and d[0] > d[2]:
                canal[n-1][m-1] = 0
            elif d[2] > d[1] and d[2] > d[0]:
                canal[n-1][m-1] = 2
            else:
                canal[n-1][m-1] = 1

            
    
    valoresConversion = [lbpB,lbpG,lbpR, lbpU,lbpF, canal, sonidoPorPixelI, sonidoPorPixelF, sonidoPorPixelM]
    print(lbpF)
    print("el toro")
    print(lbpU)

    cv2.imwrite("recursosImg/lpbs/lbpB.jpg", lbpB)
    cv2.imwrite("recursosImg/lpbs/lbpG.jpg", lbpG)
    cv2.imwrite("recursosImg/lpbs/lbpR.jpg", lbpR)
    cv2.imwrite("recursosImg/lpbs/lbpU.jpg", lbpU)
    cv2.imwrite("recursosImg/lpbs/lbpF.jpg", lbpF)
    cv2.imwrite("recursosImg/viewBW.jpg", view)

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

def obtenerSonidoDeImagen(valoresConversion, x, y, numSeg, tipo):
    sonidoF = []
    print(x)
    print(y)
    canal = valoresConversion[5]
    cv2.waitKey(0)
    sonidoPorPixelM = valoresConversion[8]

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,channels=1, rate=44100, output=1)
    if tipo == "F":
        for n in list(range(y)):
            for m in list(range(x)):            
                print(n)
                print(m)
                print(sonidoPorPixelM[n][m])
                #play_tone(stream, sonidoPorPixelM[n][m],numSeg/(y*x))
                print(canal[n][m])

                senial = onda(sonidoPorPixelM[n][m],numSeg/(y*x))
                senial2 = onda(40,numSeg/(y*x))

                if canal[n][m] == 0 :
                    senial_stereo = np.ravel(np.column_stack((senial,senial2)))
                elif canal[n][m] == 1:
                    senial_stereo = np.ravel(np.column_stack((senial,senial)))
                else:
                    senial_stereo = np.ravel(np.column_stack((senial2,senial)))

                reproducir(stream,senial_stereo )                
    else:
        n = -3
        while n < y:
            n = n+3
            if n <= y-1: 
                m = -3               
                while m < x:
                    m = m+3
                    if m <= x-1:
                        print(n)
                        print(m)
                        print(sonidoPorPixelM[n][m])
                        print(canal[n][m])

                        senial = onda(sonidoPorPixelM[n][m],2*numSeg/(y*x))
                        senial2 = onda(40,2*numSeg/(y*x))

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

def inicio(nombreImagen, numSeg, x, y, tipo):
    img = mostrarImagenInicialEstandard(nombreImagen, x,y)
    bgr = obtenerMatricesBGR(img, x,y)
    valoresConversion= obtenerValoresConversion(img,bgr,x,y)

    sonidoDeImagen = obtenerSonidoDeImagen(valoresConversion, x-2, y-2, numSeg, tipo)
    cv2.waitKey(0)

#inicio("srcImagenes/carito.jpg", 60, 200,150, "S")
inicio("srcImagenes/banderaEcu.jpg", 30, 400,300, "F")

print("Graciassss TOTALES!!")
