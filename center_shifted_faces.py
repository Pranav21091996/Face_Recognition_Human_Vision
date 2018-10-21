import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage


def get_immediate_subdirectories(a_dir):
	return [name for name in os.listdir(a_dir)
			if os.path.isdir(os.path.join(a_dir, name))]

binarymask = 'BinaryMask'
facemask = 'FaceMask'
CroppedFolder1 = "placemaskleft"
CroppedFolder2 = "placemaskright"
CroppedFolder3 = "placemaskup"
CroppedFolder4 = "placemaskdown"
shift = 16

folder = [CroppedFolder1,CroppedFolder2,CroppedFolder3,CroppedFolder4]
for CroppedFolder in folder:
	if not os.path.exists(CroppedFolder):
			os.makedirs(CroppedFolder)

BLACK = [0,0,0]
left_border = [96-shift,96+shift,96,96]
right_border = [96+shift,96-shift,96,96]
up_border = [96,96,96-shift,96+shift]
down_border = [96,96,96+shift,96-shift]
size = [64,64,64,64]


BLACK = [0,0,0]
WHITE = [255,255,255]

for directory in get_immediate_subdirectories(binarymask):
	for CroppedFolder in folder:
		if not os.path.exists(os.path.join(CroppedFolder, directory)):
			os.makedirs(os.path.join(CroppedFolder, directory))
	for filename in os.listdir(os.path.join(binarymask,directory)):
		try:
			for k in range(4):
				mask = plt.imread(os.path.join(binarymask,directory,filename))
				face_mask = plt.imread(os.path.join(facemask,directory,filename))
				backg = plt.imread('green.jpg')
				backg = cv2.resize(backg,(256,256))
				mask = cv2.resize(mask,(size[k],size[k]))
				face_mask = cv2.resize(face_mask,(size[k],size[k]))
				mask = cv2.bitwise_not(mask)
				leftDiff = left_border[k]
				rightDiff = right_border[k]
				downDiff = down_border[k]
				upDiff = up_border[k]
				mask = cv2.copyMakeBorder(mask,upDiff,downDiff, leftDiff,rightDiff,cv2.BORDER_CONSTANT, value=WHITE )
				face_mask = cv2.copyMakeBorder(face_mask,upDiff,downDiff, leftDiff,rightDiff,cv2.BORDER_CONSTANT, value=BLACK )
				img3 = cv2.bitwise_and(backg,mask)
				img5 = cv2.add(face_mask,img3)
				plt.imsave(os.path.join(folder[k],directory,filename),img5)
		except Exception as e:
			print(e)
