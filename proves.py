# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:34:47 2020

@author: carles
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

im=cv2.imread('comp3.jpg')
R=im[:,:,2]
G=im[:,:,1]
B=im[:,:,0]

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:(6*25):25,0:(9*25):25].T.reshape(-1,2)



_img_shape = None

images = glob.glob('*.jpg')
for fname in images:
    img = cv2.imread(fname)       
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners= cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH  + cv2.CALIB_CB_NORMALIZE_IMAGE)  #+ cv2.CALIB_CB_FAST_CHECK
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (20,20),(-1,-1), criteria)
        imgpoints.append(corners)

ret, intrinsic_matrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print ('matriu intrinseca K=',intrinsic_matrix,'coeficients de distorció: D=', distCoeff)


newMat, ROI = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distCoeff, gray.shape[::-1], alpha = 0.5, centerPrincipalPoint = 1)


mapx, mapy = cv2.initUndistortRectifyMap(intrinsic_matrix, distCoeff, None, intrinsic_matrix, gray.shape[::-1], m1type = cv2.CV_32FC1)


Rd = cv2.remap(R, mapx, mapy, cv2.INTER_NEAREST)
Gd = cv2.remap(G, mapx, mapy, cv2.INTER_LINEAR)
Bd = cv2.remap(B, mapx, mapy, cv2.INTER_LINEAR)
dst=np.zeros([1944,2592,3],np.uint8)
dst[:,:,0]=np.uint8(Rd)
dst[:,:,1]=np.uint8(Gd)
dst[:,:,2]=np.uint8(Bd)
plt.figure('2')
plt.imshow(dst)

a=np.array(rvecs[0])
b=np.array(tvecs[0])
c=cv2.Rodrigues(a)[0]
P_mat=np.dot(intrinsic_matrix,np.concatenate((c,b),axis=1))

fx=1133.62
fy=1122.9
cx=1260
cy=922
k1=distCoeff[0,0]
k2=distCoeff[0,1]
k3=distCoeff[0,4]
p1=distCoeff[0,2]
p2=distCoeff[0,3]
R=cv2.Rodrigues(a)[0]
map_x=np.zeros((1944,2592))
map_y=np.zeros((1944,2592))

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
        
im22=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
prova=cv2.remap(im22,map_x,map_y,cv2.INTER_LINEAR)
plt.imshow(prova,'gray')
        '''
        if im_nova[pix_x,pix_y,0].any()==0:
            im_nova[(pix_x),(pix_y),0]=im[(pix_xnou),(pix_ynou),0]
        else:
           im_nova[pix_x,pix_y,0]=(im_nova[pix_x,pix_y,0]+im[pix_xnou,pix_ynou,0])/2
           '''


'''
if found == True:

    #Add the "true" checkerboard corners
    opts.append(objp)

    #Improve the accuracy of the checkerboard corners found in the image and save them to the ipts variable.
    cv2.cornerSubPix(gray, corners, (20, 20), (-1, -1), criteria)
    ipts.append(corners)

    #Draw chessboard corners
    cv2.drawChessboardCorners(im, (9,6), corners, found)

    #Show the image with the chessboard corners overlaid.
   # cv2.imshow("Corners", im)
   '''




'''
K = np.array([[  857,     0.  , 300],
              [    0.  ,   876,   556.17],
              [    0.  ,     0.  ,     1.  ]])

# zero distortion coefficients work well for this image
D = np.array([-0.257614, 0.087708, -0.00025697, -0.015219])

# use Knew to scale the output
Knew = K.copy()
Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (800,600), cv2.CV_16SC2)
nemImg = cv2.remap( img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
#cv2.imshow('undistorted', img_undistorted)
cv2.imshow('img',nemImg)
'''