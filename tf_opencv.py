import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, GRU, Embedding , Flatten, Dropout
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import load_model
import memory_profiler as mem_profile
import time

import cv2

print("Imports successful") 

# Load model from folder
try:
	# load model
	model = load_model('model1.h5')
	print("model loaded")
except:
	print("impossible del oader le modele ")
# summarize model.
#model.summary()

#create object, 0 for external cam
video = cv2.VideoCapture(0)

#Display fps of webcam 
print(video.get(cv2.CAP_PROP_FPS))

frames = 0
font = cv2.FONT_HERSHEY_SIMPLEX
BOX=True

mem_before=mem_profile.memory_usage()

def boxing(img):
	image, contours, hier = cv2.findContours(img, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	k=0
	mem=img
	w_total=[]
	global list_box
	list_box=[]
	list_box_position=[]
	global pos
	pos = []
	for c in contours:
		if k<len(contours)-1:                    #enleve le gros cadre k<len(contours)-1:
		  # get the bounding rect
			x, y, w, h = cv2.boundingRect(c)
			w_total.append(w)
			
			# cv2.rectangle(output, (x, y), (x + w, y + h), (0,255,0), 2)#1
			# k=k+1
			# pos.append([x, y])
			# list_box.append(mem[y:y+h,x:x+w])
			
			if h>48 and w>48:
			#draw a white rectangle to visualize the bounding rect
				cv2.rectangle(output, (x, y), (x + w, y + h), (0,255,0), 1)
				
				pos.append([x, y])
				list_box.append(mem[y:y+h,x:x+w])
			k=k+1
	#       list_box_position.append([[x:x+w], [y:y+h])
	# cv2.drawContours(output, contours, -1, (255, 0, 0), 1)
	
def process_frame(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow("FRAME",frame)
	
	frame_cropped = frame[0:480,80:560]
	smallest = frame_cropped.min(axis=0).min(axis=0)
	biggest = frame_cropped.max(axis=0).max(axis=0)	
	meann = cv2.mean(frame_cropped)
	print("_____________________ small = {} et big = {} , mean = {}".format(smallest, biggest, meann) )
	cv2.threshold(frame_cropped,90,255,cv2.THRESH_BINARY,frame_cropped) #binary+cv2.THRESH_OTSU
	#frame_cropped = cv2.adaptiveThreshold(frame_cropped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	kernel = np.ones((5,5),np.uint8)
	frame_eroded = cv2.erode(frame_cropped,kernel,1)
	
	global output 
	output = cv2.cvtColor(frame_eroded,cv2.COLOR_GRAY2RGB)
	
	frame_eroded = ~frame_eroded
	frame_resized = frame_eroded
	
	found = False 
	
	if BOX!=True:
		frame_resized = cv2.resize(frame_eroded, (28,28)) #remplacable par "reshape" ?
		
		#print("le max de l'image brute est {} !".format(max(frame_reshaped)))
	else:
		boxing(frame_eroded)

		if len(list_box)!=0 and len(list_box[-1]>10):
			# find longest box (x axis) -> to be improved?
			list_lengths= [len(x) for x in list_box]
			k=np.argmax(list_lengths)
			print(list_lengths)
			box = list_box[k]
			shape=box.shape
			padding = int(abs((shape[0]-shape[1])/2))
			pad_top = int(shape[0]/8)
			frame_resized = cv2.copyMakeBorder(box, top=pad_top, bottom=pad_top, left=padding+pad_top, right=padding+pad_top, borderType=cv2.BORDER_CONSTANT)
			frame_resized = cv2.resize(frame_resized, (28,28))
			cv2.imshow("liste box {}", frame_resized)
			print("frame resized {} (box valide)".format(frame_resized.shape))
			found=True
			frame_reshaped = frame_resized.reshape(1, 28, 28, 1)	
			predicted = model.predict(frame_reshaped)
			predicted_int = np.argmax(predicted, axis=1)
			print(predicted_int)
			# else:
				# frame_resized = cv2.resize(frame_eroded, (28,28)) #oblige sinon en cas dexcept il est perdu apres

			## exepct
		else:
			#frame_resized = cv2.resize(frame_eroded, (28,28)) #oblige sinon en cas dexcept il est perdu apres
			print("NO box detected")
	
		
	#frame_normalized = frame_resized/255#cv2.normalize(frame_resized, 0, 255)
	
	
	
	
	if found:
		cv2.putText(output,str(predicted_int),tuple(pos[k]), font, 3,(0,0,255),3,cv2.LINE_AA)
	cv2.imshow("capture cropped", output)
	
		
		
		
	#predict_frame(frame_reshaped)
	
	#print(frame_normalized)
	#cv2.imwrite("reshaped.png",frame_resized)

	

t0 = time.clock()

while True:
	t1 = time.clock()
	frames = frames +1
	check,frame = video.read()
	process_frame(frame)
	key = cv2.waitKey(1)
	t2 = time.clock()
	print("Temps ecoule {}".format((t2 - t1)) )

	
	if key == ord('q') or key == ord('Q'):
		break
t3 = time.clock()
print("Temps ecoule par frame {}".format((t3 - t0)/frames) )
print("End in {} images".format(time))
# shutdown camera
video.release()
mem_after = mem_profile.memory_usage()
print("Memory used : {}Mb ". format(mem_after[0] - mem_before[0]) ) 

	
def predict_frame(frame_processed):
	predicted = model.predict(frame_processed)
	print(np.argmax(predicted, axis=1))