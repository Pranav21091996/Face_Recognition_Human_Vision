from utils import *


dataSet_directory1 = sys.argv[1]
dataSet_directory2 =sys.argv[2]
dataSet_directory3 =sys.argv[3]
dataSet_directory4 = sys.argv[4]
dataSet_directory5 = sys.argv[5]

Dataset1_Input = []
Dataset1_Label = []
Dataset2_Input = []
Dataset2_Label = []
Dataset3_Input = []
Dataset3_Label = []
Dataset4_Input = []
Dataset4_Label = []
Dataset5_Input = []
Dataset5_Label = []


num_crops = 4
img_size = 64
i=0

for ImageDir in os.listdir(dataSet_directory1):
	index=(32,64,128,256)
	image_directory1 = dataSet_directory1+'/'+ImageDir
	image_directory2 = dataSet_directory2+'/'+ImageDir
	image_directory3 = dataSet_directory3+'/'+ImageDir
	image_directory4 = dataSet_directory4+'/'+ImageDir
	image_directory5 = dataSet_directory5+'/'+ImageDir

	print(image_directory1,image_directory2,image_directory3,image_directory4,image_directory5)
	print(i)
	
	for image in os.listdir(image_directory1):
		list = []
		list1 = []
		try:
			im = cv2.imread(image_directory1+'/'+image)
			im = cv2.resize(im, (256,256))

			for crop_index in range(num_crops):
				imageToCrop=im
				resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
				resizedImage= normalize(resizedImage)
				list.append(resizedImage)
			stackedImage=np.vstack((list))
			stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
			Dataset1_Input.append(stackedImage)
			Dataset1_Label.append(i)
			
			im2 = cv2.imread(image_directory2+'/'+image)
			im2 = cv2.resize(im2, (256,256))
			for crop_index in range(num_crops):
				imageToCrop=im2
				resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
				resizedImage= normalize(resizedImage)
				list1.append(resizedImage)
			stackedImage=np.vstack((list1))
			list1 = []
			stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
			Dataset2_Input.append(stackedImage)
			Dataset2_Label.append(i)

			im3 = cv2.imread(image_directory3+'/'+image)
			im3 = cv2.resize(im3, (256,256))
			for crop_index in range(num_crops):
				imageToCrop=im3
				resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
				resizedImage= normalize(resizedImage)
				list1.append(resizedImage)
			stackedImage=np.vstack((list1))
			list1 = []
			stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
			Dataset3_Input.append(stackedImage)
			Dataset3_Label.append(i)

			im4 = cv2.imread(image_directory4+'/'+image)
			im4 = cv2.resize(im4, (256,256))
			for crop_index in range(num_crops):
				imageToCrop=im4
				resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
				resizedImage= normalize(resizedImage)
				list1.append(resizedImage)
			stackedImage=np.vstack((list1))
			list1 = []
			stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
			Dataset4_Input.append(stackedImage)
			Dataset4_Label.append(i)

			im5 = cv2.imread(image_directory5+'/'+image)
			im5 = cv2.resize(im5, (256,256))
			for crop_index in range(num_crops):
				imageToCrop=im5
				resizedImage=crop_center_resize(imageToCrop,index[crop_index],index[crop_index])
				resizedImage= normalize(resizedImage)
				list1.append(resizedImage)
			stackedImage=np.vstack((list1))
			list1 = []
			stackedImage=stackedImage.reshape([-1,img_size,img_size,3])
			Dataset5_Input.append(stackedImage)
			Dataset5_Label.append(i)

		except Exception as e:
			print ("Unexpected error:", str(e))
	i+=1

np.save(sys.argv[1]+'_Input.npy',Dataset1_Input)
np.save(sys.argv[1]+'_Labels.npy',Dataset1_Label)
np.save(sys.argv[2]+'_Input.npy',Dataset2_Input)
np.save(sys.argv[2]+'_Labels.npy',Dataset2_Label)
np.save(sys.argv[3]+'_Input.npy',Dataset3_Input)
np.save(sys.argv[3]+'_Labels.npy',Dataset3_Label)
np.save(sys.argv[4]+'_Input.npy',Dataset4_Input)
np.save(sys.argv[4]+'_Labels.npy',Dataset4_Label)
np.save(sys.argv[5]+'_Input.npy',Dataset5_Input)
np.save(sys.argv[5]+'_Labels.npy',Dataset5_Label)
