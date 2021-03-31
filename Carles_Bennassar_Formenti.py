# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:03:48 2020

@author: carles
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

'''
IMPORTANT!

Aquest programa tarda uns 8 minuts en correr. Això es deu a que primer ha de entrenar unes quantes imatges (9 en aquest cas) per a treure 
els paràmetres geomètrics de la càmera i a dos fors que han de recorrer cada píxel de una imatge qualssevol i fer-ne diver-ses operacions 
(acada píxel)
'''


#defineixo la meva quadricula
tauler = (6,9)
#criteri de iteracions per a trobar amb exactitud on es troben els corners del taulell d'escacs
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#inicialitzo dos arrays buides objpoints que serà el número de punts que bull trobar segons el taulell que
#'sutilitzi; imgpoints on hi emmagatzemaré tots els corners trobats (imgpoints==objpoints)
objpoints = []
imgpoints=[]
#creo les coordenades pel mon real per punts 3D, amb Z=0 i de tamany: (1,tots els quadrats del tauler,3per XYZ)
objp = np.zeros((1, tauler[0] * tauler[1], 3), np.float32)
#ho format-ho en format matriu XY, obtinc "un tauler d'escacs virtual" (quadricula preparada per a posar-hi les coordenades
#dels corners trobats)
objp[0,:,:2] = np.mgrid[0:tauler[0], 0:tauler[1]].T.reshape(-1, 2)


#aquest codi serveix per a llegir tot el set d'imatges, comprovar que totes tenen el mateix format
imatges = glob.glob('*.jpg')
for fname in imatges:
    img = cv2.imread(fname)
#per a cada imatge trobo on estan els corners del meu taulell i els guardo a imgpoints        
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners= cv2.findChessboardCorners(gray, tauler,cv2.CALIB_CB_ADAPTIVE_THRESH  + cv2.CALIB_CB_NORMALIZE_IMAGE)  
    if ret == True: #ret =True si detecta corners =False si no els detecta 
        #omplo les mesves arrays amb la posició dels corners en imgpoints, i per a cada fotografia amb corners trobat creo un conjunt de coordenades
        #a objpoints
        
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (20,20),(-1,-1), criteria)
        imgpoints.append(corners)


'''
K,D i nou mapa AMB LLIBRERIA CV2.EYEFISH
'''
#aquesta funció troba K i D mitjançant els corners de totes les imatges analitzades
ret, Keye, Deye, rvecs, tvecs = cv2.fisheye.calibrate( objpoints,imgpoints, gray.shape[::-1], None, None)
print('matriu intrinseca Keye=',Keye,'distorció Deye=', Deye)
#aquesta funció troba una K més precisa
Krec=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(Keye,Deye,gray.shape[::-1],None)
# creo el mapa de de la camera estudiada per a una imatge de un tamany concret
mapxeye, mapyeye = cv2.fisheye.initUndistortRectifyMap(Keye,Deye,None,Keye,gray.shape[::-1],cv2.CV_32FC1) 



'''
K,D i nou mapa AMB LLIBRERIA CV2 per a qualssevol tipus de camera
'''
#aquestes 3 línes fan el mateix que les d'adalt amb un altre mòdul
ret, Kmtx, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

newMat, ROI = cv2.getOptimalNewCameraMatrix(Kmtx, D, gray.shape[::-1], alpha = 0.5, centerPrincipalPoint = 1)

mapxg, mapyg = cv2.initUndistortRectifyMap(Kmtx, D, None, Kmtx, gray.shape[::-1], m1type = cv2.CV_32FC1)
print ('matriu intrinseca K=',Kmtx,'coeficients de distorció: D=', D)

'''
K i D anteriors (són els que millor representan la lent) i mapa fet a mà
'''
#amb tot aquest codi el que faig és calcular a mà com fer el mapa de la càmera per una imatge de un tamany concret
fx=Kmtx[0,0]
fy=Kmtx[1,1]
cx=Kmtx[0,2]
cy=Kmtx[1,2]
k1=D[0,0]
k2=D[0,1]
k3=D[0,4]
p1=D[0,2]
p2=D[0,3]
a=np.array(rvecs[0])
R=cv2.Rodrigues(a)[0]
map_x=np.zeros((1944,2592))
map_y=np.zeros((1944,2592))

#el for a de recorrer cada píxel de la imatge, i per a cada píxel fer varies operacions, aquests fors tarden 1 o 2 minuts en correr
for u in range (0,1944):
    for v in range (0,2592):
        x=(u-cx)/fx
        y=(v-cy)/fy
        XYW=np.dot(np.linalg.inv(R),np.transpose(np.array((x,y,1))))
        XYW1=np.dot((R),np.transpose(np.array((x,y,1))))
        pix_x=XYW[0]/XYW[2]
        pix_y=XYW1[1]/XYW1[2]
        r=np.sqrt(pix_x**2+pix_y**2)
        pix_xnou=pix_x*(1+(k1*r**2)+(k2*r**4)+(k3*r**6))+ (2*p1*pix_x*pix_y)+p2*((r**2)+2*pix_x**2)
        pix_ynou=pix_y*(1+(k1*r**2)+(k2*r**4)+(k3*r**6))+ (2*p2*pix_x*pix_y)+p1*((r**2)+2*pix_y**2)
        map_x[u,v]=pix_xnou*fx+cx
        map_y[u,v]=pix_ynou*fy+cy
map_x=np.float32(map_x)
map_y=np.float32(map_y)

#aquesta funció només està per a facilitar la implementació del remap per a poder fer plots il·lustratius de cada mètode usat
def new_pic(foto,tecnica_fisheye,mapa_calculat): #true per a fer servir llibreria cv2.fisheye
    imnova=np.zeros(foto.shape,np.uint8)
    if tecnica_fisheye == True and mapa_calculat==False:
        mapx=mapxeye
        mapy=mapyeye
        #la llibreria cv2 treballa amb BGR envès de RBG per això aquest ajust en l'ordre dels colors
        #la funció remap agafa els mapes creat i els aplica a la fotografía que jo vulgui per a corregir la distorció de la imatge
        Rd = cv2.remap(foto[:,:,2], mapx, mapy, cv2.INTER_LINEAR)
        Gd = cv2.remap(foto[:,:,1], mapx, mapy, cv2.INTER_LINEAR)
        Bd = cv2.remap(foto[:,:,0], mapx, mapy, cv2.INTER_LINEAR)
    elif tecnica_fisheye == False and mapa_calculat==True:
        mapx=map_x
        mapy=map_y
        Rd = cv2.remap(foto[:,:,2], mapx, mapy, cv2.INTER_LINEAR)
        Gd = cv2.remap(foto[:,:,1], mapx, mapy, cv2.INTER_LINEAR)
        Bd = cv2.remap(foto[:,:,0], mapx, mapy, cv2.INTER_LINEAR)
    else:
        mapx=mapxg
        mapy=mapyg
        Rd = cv2.remap(foto[:,:,2], mapx, mapy, cv2.INTER_LINEAR)
        Gd = cv2.remap(foto[:,:,1], mapx, mapy, cv2.INTER_LINEAR)
        Bd = cv2.remap(foto[:,:,0], mapx, mapy, cv2.INTER_LINEAR)
    imnova[:,:,0]=np.uint8(Rd)
    imnova[:,:,1]=np.uint8(Gd)
    imnova[:,:,2]=np.uint8(Bd)
    return  imnova


im1=cv2.imread('pro3.jpg')
im11=cv2.cvtColor(im1, cv2.COLOR_BGR2RGB )
prova_eye1=new_pic(im1, True,False)
prova_gen1=new_pic(im1,False,False)
prova_ama1=new_pic(im1, False,True)
im2=cv2.imread('comp1.jpg')
im22=cv2.cvtColor(im2, cv2.COLOR_BGR2RGB )
prova_eye2=new_pic(im2, True,False)
prova_gen2=new_pic(im2,False,False)
prova_ama2=new_pic(im2, False,True)


#sublplots comparatius dels resultats obtinguts
plt.figure('Resulatats')
plt.subplot(241)
plt.title('Imatge set original')
plt.imshow(im11)
plt.subplot(242)
plt.title('Imatge set amb llibreria cv2.eyefish')
plt.imshow(prova_eye1)

plt.subplot(243)
plt.title('Imatge set amb llibreria general')
plt.imshow(prova_gen1)
plt.subplot(244)
plt.title('Imatge set amb mapa calculat')
plt.imshow(prova_ama1)

plt.subplot(245)
plt.title('Imatge test original')
plt.imshow(im22)
plt.subplot(246)
plt.title('Imatge test amb llibreria cv2.eyefish')
plt.imshow(prova_eye2)

plt.subplot(247)
plt.title('Imatge test amb llibreria general')
plt.imshow(prova_gen2)
plt.subplot(248)
plt.title('Imatge test amb mapa calculat')
plt.imshow(prova_ama2)



