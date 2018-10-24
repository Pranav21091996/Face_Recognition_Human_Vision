import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
import random
import os
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import sys
from utils import *

InputImage = np.load(sys.argv[1]+'_InputImage.npy')
Labels = np.load(sys.argv[1]+'_Labels.npy')
test_Data1 = np.load(sys.argv[1]+'_test_Data1.npy')
test_label1 = np.load(sys.argv[1]+'_test_label1.npy')

test_Data2 = np.load(sys.argv[2]+'_test_Data2.npy')
test_label2 = np.load(sys.argv[2]+'_test_label2.npy')

test_Data3 = np.load(sys.argv[3]+'_test_Data3.npy')
test_label3 = np.load(sys.argv[3]+'_test_label3.npy')
test_Data4 = np.load(sys.argv[4]+'_test_Data4.npy')
test_label4 = np.load(sys.argv[4]+'_test_label4.npy')
test_Data5 = np.load(sys.argv[5]+'_test_Data5.npy')
test_label5 = np.load(sys.argv[5]+'_test_label5.npy')

num_crops = 4
img_size = 64
i=0

TRAINING_EPOCHS = 10000
BATCH_SIZE = 98
DISPLAY_STEP = 10



for ImageDir in os.listdir(dataSet_directory):
	index=(32,64,128,256)
	image_directory = dataSet_directory+'/'+ImageDir
	
	test_image_directory2 = testData_directory2+'/'+ImageDir
	test_image_directory3 = testData_directory3+'/'+ImageDir
	test_image_directory4 = testData_directory4+'/'+ImageDir
	test_image_directory5 = testData_directory5+'/'+ImageDir
	
	print(image_directory,test_image_directory2,test_image_directory5)
	print(i)
	count=0
	for image in os.listdir(image_directory):
		list = []
		list1 = []
		try:
			im = cv2.imread(image_directory+'/'+image)
			im = cv2.resize(im, (256,256))

			for crop_index in range(num_crops):
				imageToCrop=im
				resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
				resizedImage= normalize(resizedImage)
				list.append(resizedImage)
			stackedImage=np.vstack((list))
			stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
			count+=1
			if(count<100):
				InputImage.append(stackedImage)
				Labels.append(i)
			
			else:
				test_Data1.append(stackedImage)
				test_label1.append(i)
				
				im2 = cv2.imread(test_image_directory2+'/'+image)
				im2 = cv2.resize(im2, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im2
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data2.append(stackedImage)
				test_label2.append(i)
				
				im3 = cv2.imread(test_image_directory3+'/'+image)
				im3 = cv2.resize(im3, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im3
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data3.append(stackedImage)
				test_label3.append(i)

				im4 = cv2.imread(test_image_directory4+'/'+image)
				im4 = cv2.resize(im4, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im4
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data4.append(stackedImage)
				test_label4.append(i)
				
				im5 = cv2.imread(test_image_directory5+'/'+image)
				im5 = cv2.resize(im5, (256,256))
				for crop_index in range(num_crops):
					imageToCrop=im5
					resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
					resizedImage= normalize(resizedImage)
					list1.append(resizedImage)
				stackedImage=np.vstack((list1))
				list1 = []
				stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
				test_Data5.append(stackedImage)
				test_label5.append(i)
				
		except Exception as e:
			print ("Unexpected error:", str(e))					
	i+=1	





Data = InputImage
Labels = Labels
one_hot_labels = one_hot_encode(Labels)
one_hot_labels_test1 = one_hot_encode(test_label1)
one_hot_labels_test2 = one_hot_encode(test_label2)
one_hot_labels_test3 = one_hot_encode(test_label3)
one_hot_labels_test4 = one_hot_encode(test_label4)
one_hot_labels_test5 = one_hot_encode(test_label5)

train_Data,train_label = shuffle(Data,one_hot_labels,random_state=20)
test_Data1,test_label1 = shuffle(test_Data1,one_hot_labels_test1,random_state=20)
test_Data2,test_label2 = shuffle(test_Data2,one_hot_labels_test2,random_state=20)
test_Data3,test_label3 = shuffle(test_Data3,one_hot_labels_test3,random_state=20)
test_Data4,test_label4 = shuffle(test_Data4,one_hot_labels_test4,random_state=20)
test_Data5,test_label5 = shuffle(test_Data5,one_hot_labels_test5,random_state=20)

print("size of the train data", len(Data))
print("shape of the train data", np.shape(Data))
print("size of the test data1", len(test_Data1))
print("size of the test data2", len(test_Data2))
print("size of the test data3", len(test_Data3))
print("size of the test data4", len(test_Data4))
print("size of the test data5", len(test_Data5))

VALIDATION_SIZE = len(test_Data1)

x = neural_net_image_input()
y = neural_net_label_input(100)    
keep_prob = neural_net_keep_prob_input()

logits = conv_net(x, keep_prob)
Y_pred = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32), name = 'accuracy')


predict = tf.argmax(Y_pred, 1)

epochs_completed = 0
index_in_epoch = 0
num_examples = len(train_Data)
print(num_examples)


init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

train_accuracies = []
validation_accuracies = []

DISPLAY_STEP=1

for i in range(TRAINING_EPOCHS):

    
    batch_xs, batch_ys = next_batch(BATCH_SIZE)  
          

    
    if i%DISPLAY_STEP == 0 or (i+1) == TRAINING_EPOCHS:

         train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y: batch_ys,
                                                  keep_prob: 0.5}) 
         if(VALIDATION_SIZE):  
            validation_accuracy = accuracy.eval(feed_dict={ x: test_Data1[0:BATCH_SIZE],y: test_label1[0:BATCH_SIZE],keep_prob: 0.5})
            print('training_accuracy => %.2f for step %d'%(train_accuracy , i))
         else:
            print('train_accuracicy => %.4f for step %d'%(train_accuracy, i))
            train_accuracies.append(train_accuracy)
         if i%(DISPLAY_STEP*10) == 0 and i:
            DISPLAY_STEP *=10
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data1[0:BATCH_SIZE],y: test_label1[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 1 => %.2f for step %d'%(validation_accuracy, i))
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data2[0:BATCH_SIZE],y: test_label2[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 2 => %.2f for step %d'%(validation_accuracy, i))
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data3[0:BATCH_SIZE],y: test_label3[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 3 => %.2f for step %d'%(validation_accuracy, i))
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data4[0:BATCH_SIZE],y: test_label4[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 4 => %.2f for step %d'%(validation_accuracy, i))
    validation_accuracy = accuracy.eval(feed_dict={ x: test_Data5[0:BATCH_SIZE],y: test_label5[0:BATCH_SIZE],keep_prob: 0.5})
    print('validation_accuracy 5 => %.2f for step %d'%(validation_accuracy, i))
    
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,keep_prob: 0.5})



# Below part is for getting the top 5 test accuracy for each of the five datasets
prediction1 = []
prediction2 = []
prediction3 = []
prediction4 = []
prediction5 = []

for count in range(0,len(test_Data1),BATCH_SIZE):
	if(count+BATCH_SIZE>len(test_Data1)):
		best1,best2 = sess.run([y,Y_pred],feed_dict={ x: test_Data1[count:len(test_Data1)],y: test_label1[count:len(test_Data1)],keep_prob: 0.5})
	else:
		best1,best2 = sess.run([y,Y_pred],feed_dict={ x: test_Data1[count:count+BATCH_SIZE],y: test_label1[count:count+BATCH_SIZE],keep_prob: 0.5})	
	for i in range(len(best1)):
		prediction1.append(best1[i])
		prediction1.append(best2[i])
for count in range(0,len(test_Data2),BATCH_SIZE):
	if(count+BATCH_SIZE>len(test_Data2)):
		best3,best4 = sess.run([y,Y_pred],feed_dict={ x: test_Data2[count:len(test_Data2)],y: test_label2[count:len(test_Data2)],keep_prob: 0.5})
	else:
		best3,best4 = sess.run([y,Y_pred],feed_dict={ x: test_Data2[count:count+BATCH_SIZE],y: test_label2[count:count+BATCH_SIZE],keep_prob: 0.5})	
	for i in range(len(best3)):
		prediction2.append(best3[i])
		prediction2.append(best4[i])
for count in range(0,len(test_Data3),BATCH_SIZE):
	if(count+BATCH_SIZE>len(test_Data3)):
		best5,best6 = sess.run([y,Y_pred],feed_dict={ x: test_Data3[count:len(test_Data3)],y: test_label3[count:len(test_Data3)],keep_prob: 0.5})
	else:
		best5,best6 = sess.run([y,Y_pred],feed_dict={ x: test_Data3[count:count+BATCH_SIZE],y: test_label3[count:count+BATCH_SIZE],keep_prob: 0.5})	
	for i in range(len(best5)):
		prediction3.append(best5[i])
		prediction3.append(best6[i])
for count in range(0,len(test_Data4),BATCH_SIZE):
	if(count+BATCH_SIZE>len(test_Data4)):
		best7,best8 = sess.run([y,Y_pred],feed_dict={ x: test_Data4[count:len(test_Data4)],y: test_label4[count:len(test_Data4)],keep_prob: 0.5})
	else:
		best7,best8 = sess.run([y,Y_pred],feed_dict={ x: test_Data4[count:count+BATCH_SIZE],y: test_label4[count:count+BATCH_SIZE],keep_prob: 0.5})	
	for i in range(len(best7)):
		prediction4.append(best7[i])
		prediction4.append(best8[i])
for count in range(0,len(test_Data5),BATCH_SIZE):
	if(count+BATCH_SIZE>len(test_Data5)):
		best9,best10 = sess.run([y,Y_pred],feed_dict={ x: test_Data5[count:len(test_Data5)],y: test_label5[count:len(test_Data5)],keep_prob: 0.5})
	else:
		best9,best10 = sess.run([y,Y_pred],feed_dict={ x: test_Data5[count:count+BATCH_SIZE],y: test_label5[count:count+BATCH_SIZE],keep_prob: 0.5})	
	for i in range(len(best9)):
		prediction5.append(best9[i])
		prediction5.append(best10[i])
  


f1 = open('prediction1','w')
f1.write(str(prediction1))
f2 = open('prediction2','w')
f2.write(str(prediction2))
f3 = open('prediction3','w')
f3.write(str(prediction3))
f4 = open('prediction4','w')
f4.write(str(prediction4))
f5 = open('prediction5','w')
f5.write(str(prediction5))




prediction_file = ['prediction1','prediction2','prediction3','prediction4','prediction5']
prediction_outfile = ['prediction11','prediction22','prediction33','prediction44','prediction55']
delete_list = ["[array([", ".,", ".],","dtype=float32), array([",",",'dtype=float32)]',"]",'array([0']

for infile,outfile in zip(prediction_file,prediction_outfile):
	fin = open(infile)
	fout = open(outfile, "w+")
	for line in fin:
		line = line.replace(delete_list[0], "")
		line = line.replace(delete_list[1], " ")
		line = line.replace(delete_list[2], " >")
		line = line.replace(delete_list[3], "")
		line = line.replace(delete_list[4], " ")
		line = line.replace(delete_list[5], "")
		line = line.replace(delete_list[6], " >")
		line = line.replace(delete_list[7], " ")
		fout.write(line)
	fin.close()
	fout.close()


suffixes = {0:dataSet_directory,1:testData_directory2,2:testData_directory3,3:testData_directory4,4:testData_directory5}
k=0
for outfile in prediction_outfile:
	
	import numpy as np 
	
	fp = open(outfile)
	lines = fp.readlines()
	labels = []
	prediction = []
	count = 0
	def topNAccuracy(sorted_labels,sorted_predictions,N):
		correct = 0
		for i in range(len(sorted_predictions)):
			for j in range(N):
				if(sorted_labels[i][0]==sorted_predictions[i][j]):
					correct+=1
					break
		return correct	
	list = []			
	for line in lines:
	
		for x in line.split():
			if(x=='>'):
				if(count%2==0):
					labels.append(list)
					list = []
				else:
					prediction.append(list)	
					list = []	
				
				count+=1
			else:
				list.append(float(x))
	
	
	
	sorted_labels = []	
	sorted_predictions = []
	for i in range(len(labels)):	
		sorted_labels.append(np.argsort(labels[i])[::-1][:len(labels[0])])
		sorted_predictions.append(np.argsort(prediction[i])[::-1][:len(prediction[0])])
	Accuracy = topNAccuracy(sorted_labels,sorted_predictions,1)
	print("\nprediction for",suffixes[k])
	k+=1
	print("top1 test Accuracy->",Accuracy/len(sorted_predictions)*100,"%")
	Accuracy = topNAccuracy(sorted_labels,sorted_predictions,2)
	print("top2 test Accuracy->",Accuracy/len(sorted_predictions)*100,"%")
	Accuracy = topNAccuracy(sorted_labels,sorted_predictions,3)
	print("top3 test Accuracy->",Accuracy/len(sorted_predictions)*100,"%")
	Accuracy = topNAccuracy(sorted_labels,sorted_predictions,4)
	print("top4 test Accuracy->",Accuracy/len(sorted_predictions)*100,"%")
	Accuracy = topNAccuracy(sorted_labels,sorted_predictions,5)
	print("top5 test Accuracy->",Accuracy/len(sorted_predictions)*100,"%")
	fp.close()
	



