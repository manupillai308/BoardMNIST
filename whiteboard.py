import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDARYINC = 5
IMAGESAVE = False
MODEL = load_model("./model/mnist_lenet-1-improved.h5")
PREDICT = True
LABELS = {0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine"}

pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("WhiteBoard")
#DISPLAYSURF.fill(BLACK)
iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1
while True:
	for event in pygame.event.get():
		if event.type == QUIT:
			pygame.quit()
			sys.exit()
			
		if event.type == MOUSEMOTION and iswriting:
			xcord, ycord = event.pos
			pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
			number_xcord.append(xcord)
			number_ycord.append(ycord)
			
		if event.type == MOUSEBUTTONDOWN:
			iswriting = True
			
		if event.type == MOUSEBUTTONUP:
			iswriting = False
			number_xcord = sorted(number_xcord)
			number_ycord = sorted(number_ycord)
			rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDARYINC)
			rect_min_y, rect_max_y = max(0, number_ycord[0]-BOUNDARYINC), min(number_ycord[-1]+BOUNDARYINC, WINDOWSIZEY)
			number_xcord = []
			number_ycord = []
			img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
			if IMAGESAVE:
				cv2.imwrite("images/image-{%d}.png" % image_cnt, img_arr)
				image_cnt+=1
			if PREDICT:
				image = cv2.resize(img_arr, (28, 28))
				image = np.pad(image, (10, 10), 'constant', constant_values=0)
				image = cv2.resize(image, (28, 28))
				label = str(LABELS[np.argmax(MODEL.predict(image.reshape((1,28,28,1))))]).title()
				textSurfaceObj = FONT.render(label, True, RED, WHITE)
				textRectObj = textSurfaceObj.get_rect()
				textRectObj.left, textRectObj.bottom = rect_min_x, rect_min_y
				DISPLAYSURF.blit(textSurfaceObj, textRectObj)
			pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y,
			 rect_max_x - rect_min_x, rect_max_y - rect_min_y), 3)
			
			
		if event.type == KEYDOWN:
			if event.unicode == "n":
				DISPLAYSURF.fill(BLACK)
	pygame.display.update()

