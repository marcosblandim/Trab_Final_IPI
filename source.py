# Generalizar o codigo (trocar img1 e img2 por img 1 e 2).
# Passar as imagens com argv.
# Só funciona para imagens com 1 rosto sem óculos.
# Tirar testa -> diminuir rosto até a altura do olho mais alto.
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

windows = False

# Declarar funções cascada.
if windows:
  face_cascade = cv2.CascadeClassifier('/Users/Pedro/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier('/Users/Pedro/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_eye.xml')
  smile_cascade = cv2.CascadeClassifier('/Users/Pedro/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_smile.xml')
else:
  haarcascades_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
  face_cascade = cv2.CascadeClassifier(haarcascades_path)
  haarcascades_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_eye.xml"
  eye_cascade = cv2.CascadeClassifier(haarcascades_path)
  haarcascades_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_smile.xml"
  smile_cascade = cv2.CascadeClassifier(haarcascades_path)
# Ler imagens.
img2_original= cv2.imread('faces/test2.jpg')
img1_original = cv2.imread('faces/test1.jpg')
img1 = img1_original
img2 = img2_original

# Passar para tons de cinza.
img1_gray = cv2.cvtColor(img1_original, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2_original, cv2.COLOR_BGR2GRAY)

# Região do rosto.
img1_face = (face_cascade.detectMultiScale(img1_gray, 1.3, 5))[0] # Image, scaling factor, minNeighbours.
img2_face = (face_cascade.detectMultiScale(img2_gray))[0] # img,1.3,5

# ROI's da face
roi1 = img1_gray[img1_face[1]:(img1_face[1]+img1_face[3]),img1_face[0]:(img1_face[0]+img1_face[2])]
roi2 = img2_gray[img2_face[1]:(img2_face[1]+img2_face[3]),img2_face[0]:(img2_face[0]+img2_face[2])]

# Região dos olhos na ROI.
img1_eyes = (eye_cascade.detectMultiScale(roi1))[0:2] # 1.1,5
img2_eyes = (eye_cascade.detectMultiScale(roi2))[0:2]

# Inclinação.
(x1,y1,w1,h1) = img1_eyes[1]
(x2,y2,w2,h2) = img1_eyes[0]
ang1 = int(math.degrees(math.atan((y2+(h2/2))-(y1+(h1/2))/((x2+(w2/2))-(x1+(w1/2)))))) - 90
(x1,y1,w1,h1) = img2_eyes[1]
(x2,y2,w2,h2) = img2_eyes[0]
ang2 = int(math.degrees(math.atan((y2+(h2/2))-(y1+(h1/2))/((x2+(w2/2))-(x1+(w1/2)))))) - 90

# Rotacionar.
matriz_rotacao = cv2.getRotationMatrix2D((int(img1.shape[1]/2),int(img1.shape[0]/2)),ang1,1)
img1 = cv2.warpAffine(img1,matriz_rotacao,(img1.shape[1],img1.shape[0]))
matriz_rotacao = cv2.getRotationMatrix2D((int(img2.shape[1]/2),int(img2.shape[0]/2)),ang2,1)
img2 = cv2.warpAffine(img2,matriz_rotacao,(img2.shape[1],img2.shape[0]))

# Atuliazacao pos rotacao.

# Passar para tons de cinza.
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Região do rosto.
img1_face = (face_cascade.detectMultiScale(img1_gray, 1.3, 5))[0] # Image, scaling factor, minNeighbours.
img2_face = (face_cascade.detectMultiScale(img2_gray, 1.3, 5))[0] # img,1.3,5

# ROI's da face
roi1 = img1_gray[img1_face[1]:(img1_face[1]+img1_face[3]),img1_face[0]:(img1_face[0]+img1_face[2])]
roi2 = img2_gray[img2_face[1]:(img2_face[1]+img2_face[3]),img2_face[0]:(img2_face[0]+img2_face[2])]

# Região dos olhos na ROI.
img1_eyes = (eye_cascade.detectMultiScale(roi1))[0:2] # 1.1,5
img2_eyes = (eye_cascade.detectMultiScale(roi2,1.2,3))[0:2]

# Região do sorriso na ROI.
img1_smile = (smile_cascade.detectMultiScale(roi1,1.8,4))[0] # 1.8, 5
img2_smile = (smile_cascade.detectMultiScale(roi2,1.8,4))[0] # 1.8, 4

# Marcar ROI.
for (x,y,w,h) in (img1_face,[0,0,0,0]):
    cv2.rectangle(img1_original,(x,y),(x+w,y+h),(255,0,0),3)
for (x,y,w,h) in (img2_face,[0,0,0,0]):
    cv2.rectangle(img2_original,(x,y),(x+w,y+h),(255,0,0),3)
for (ex,ey,ew,eh) in img1_eyes:
    cv2.rectangle(roi1,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
for (ex,ey,ew,eh) in img2_eyes:
    cv2.rectangle(roi2,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
for (x,y,w,h) in (img1_smile,[0,0,0,0]):
    cv2.rectangle(roi1,(x,y),(x+w,y+h),(255,0,0),3)
for (x,y,w,h) in (img2_smile,[0,0,0,0]):
    cv2.rectangle(roi2,(x,y),(x+w,y+h),(255,0,0),3)

# Diminuir img_face.
#img1_face[2] = max(img1_eyes[0,0]+img1_eyes[0,2],img1_eyes[1,0]+img1_eyes[1,2]) - min(img1_eyes[0,0],img1_eyes[1,0])
img1_face[2] = int(1.1*(max(img1_eyes[0,0],img1_eyes[1,0]) - min(img1_eyes[0,0],img1_eyes[1,0]) + min(img1_eyes[0,2],img1_eyes[1,2])))
img1_face[3] = int(0.87*(img1_smile[1] + img1_smile[3] - min(img1_eyes[0,0],img1_eyes[1,0])))
img1_face[0] += int(0.9*(min(img1_eyes[0,0],img1_eyes[1,0])))
img1_face[1] += min(img1_eyes[0,1],img1_eyes[1,1])

#img2_face[2] = max(img2_eyes[0,0]+img2_eyes[0,2],img2_eyes[1,0]+img2_eyes[1,2]) - min(img2_eyes[0,0],img2_eyes[1,0])
img2_face[2] = int(1.1*(max(img2_eyes[0,0],img2_eyes[1,0]) - min(img2_eyes[0,0],img2_eyes[1,0]) + min(img2_eyes[0,2],img2_eyes[1,2])))
img2_face[3] = int(0.87*(img2_smile[1] + img2_smile[3] - min(img2_eyes[0,0],img2_eyes[1,0])))
img2_face[0] += int(0.9*(min(img2_eyes[0,0],img2_eyes[1,0])))
img2_face[1] += min(img2_eyes[0,1],img2_eyes[1,1])

# Atualizar ROI's.
roi1 = img1[img1_face[1]:(img1_face[1]+img1_face[3]),img1_face[0]:(img1_face[0]+img1_face[2])]
roi2 = img2[img2_face[1]:(img2_face[1]+img2_face[3]),img2_face[0]:(img2_face[0]+img2_face[2])]

# Escalar roi1.
roi1 = cv2.resize(roi1, (roi2.shape[1],roi2.shape[0]), interpolation = cv2.INTER_AREA)

# Processar roi1.

k=4
yuv = cv2.cvtColor(roi1,cv2.COLOR_BGR2YCrCb)
# Defino os critérios máximo de iterações = 10 e epsilon = 1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
# Transformo as 3 componentes do frame em uma imagem só YUV
# Retiro somente as crominâncias
Z = yuv[:,:,1:3].copy()
# Transformo a matriz em um vetor 1D com os canais U e V separados
Z = Z.reshape((-1, 2))
# Converto para float32
Z = np.float32(Z)
# Realizo Kmeans a partir das componentes U e V
_,labels,_ = cv2.kmeans(Z,k,None,criteria,10,flags)
labels = labels.reshape(yuv.shape[:2])
plt.matshow(labels.reshape(yuv.shape[:2]))
label = labels[labels.shape[0]//2, labels.shape[1]//2]
labels[labels==label] = -1
labels[labels!=-1] = 0
labels[labels==-1] = 1
plt.matshow(labels.reshape(yuv.shape[:2]))
# roi1 = np.uint8(roi1*labels[:,:,np.newaxis])

roi1hsv = cv2.cvtColor(roi1,cv2.COLOR_BGR2HSV)
roi2hsv = cv2.cvtColor(roi2,cv2.COLOR_BGR2HSV)
skin = np.uint8(roi2hsv*labels[:,:,np.newaxis])
skin[:,:,2:3] = 0
aux = roi1hsv.copy()
aux[:,:,0:2] = 0
noskin = np.uint8(roi1hsv*(1-labels[:,:,np.newaxis]) + aux*labels[:,:,np.newaxis])
roi1hsv = noskin + skin
roi1 = cv2.cvtColor(roi1hsv, cv2.COLOR_HSV2BGR)

# Aplicar roi1 em img2.
img2[img2_face[1]:(img2_face[1]+img2_face[3]),img2_face[0]:(img2_face[0]+img2_face[2])] = roi1

# Voltar rotacao de img 2.
matriz_rotacao = cv2.getRotationMatrix2D((int(img2.shape[1]/2),int(img2.shape[0]/2)),-ang2,1)
img2 = cv2.warpAffine(img2,matriz_rotacao,(img2.shape[1],img2.shape[0]))

# Mostrar imagens e fechar.
cv2.imshow('img1', img1_original)
cv2.imshow('img2', img2_original)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('Face 1', roi1)
cv2.imshow('Face 2', roi2)

plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
