from utils import *


dataSet_directory = sys.argv[1]
testData_directory2 =sys.argv[2]
testData_directory3 =sys.argv[3]
testData_directory4 = sys.argv[4]
testData_directory5 = sys.argv[5]

InputImage = []
Labels = []
test_Data1 = []
test_label1 = []

test_Data2 = []
test_label2 = []

test_Data3 = []
test_label3 = []
test_Data4 = []
test_label4 = []
test_Data5 = []
test_label5 = []

num_crops = 4
img_size = 64
i=0

for ImageDir in os.listdir(dataSet_directory):
	index=(32,64,128,256)
	image_directory = dataSet_directory+'/'+ImageDir

	test_image_directory2 = testData_directory2+'/'+ImageDir
	test_image_directory3 = testData_directory3+'/'+ImageDir
	test_image_directory4 = testData_directory4+'/'+ImageDir
	test_image_directory5 = testData_directory5+'/'+ImageDir

	print(image_directory,test_image_directory2,test_image_directory3,test_image_directory4,test_image_directory5)
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

np.save(sys.argv[1]+'_InputImage.npy',InputImage)
np.save(sys.argv[1]+'_Labels.npy',Labels)
np.save(sys.argv[1]+'_test_Data1.npy',test_Data1)
np.save(sys.argv[1]+'_test_label1.npy',test_label1)
np.save(sys.argv[2]+'_test_Data2.npy',test_Data2)
np.save(sys.argv[2]+'_test_label2.npy',test_label2)
np.save(sys.argv[3]+'_test_Data3.npy',test_Data3)
np.save(sys.argv[3]+'_test_label3.npy',test_label3)
np.save(sys.argv[4]+'_test_Data4.npy',test_Data4)
np.save(sys.argv[4]+'_test_label4.npy',test_label4)
np.save(sys.argv[5]+'_test_Data5.npy',test_Data5)
np.save(sys.argv[5]+'_test_label5.npy',test_label5)
