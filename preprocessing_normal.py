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


i=0

for ImageDir in os.listdir(dataSet_directory1):
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
			im = cv2.imread(image_directory3+'/'+image)
			im = cv2.resize(im, (256,256))
			im= normalize(im)
			Dataset3_Input.append(im)
			Dataset3_Label.append(i)
			
			im2 = cv2.imread(image_directory2+'/'+image)
			im2 = cv2.resize(im2, (256,256))
			im2 = normalize(im2)
			Dataset2_Input.append(im2)
			Dataset2_Label.append(i)

			im3 = cv2.imread(image_directory3+'/'+image)
			im3 = cv2.resize(im3, (256,256))
			im3 = normalize(im3)
			Dataset3_Input.append(im3)
			Dataset3_Label.append(i)

			im4 = cv2.imread(image_directory4+'/'+image)
			im4 = cv2.resize(im4, (256,256))
			im4 = normalize(im4)
			Dataset4_Input.append(im4)
			Dataset4_Label.append(i)

			im5 = cv2.imread(image_directory5+'/'+image)
			im5 = cv2.resize(im5, (256,256))
			im5 = normalize(im5)
			Dataset5_Input.append(im5)
			Dataset5_Label.append(i)

		except Exception as e:
			print ("Unexpected error:", str(e))
	i+=1


np.save(sys.argv[1]+'_Normal_Input.npy',Dataset1_Input)
np.save(sys.argv[1]+'_Normal_Labels.npy',Dataset1_Label)
np.save(sys.argv[2]+'_Normal_Input.npy',Dataset2_Input)
np.save(sys.argv[2]+'_Normal_Labels.npy',Dataset2_Label)
np.save(sys.argv[3]+'_Normal_Input.npy',Dataset3_Input)
np.save(sys.argv[3]+'_Normal_Labels.npy',Dataset3_Label)
np.save(sys.argv[4]+'_Normal_Input.npy',Dataset4_Input)
np.save(sys.argv[4]+'_Normal_Labels.npy',Dataset4_Label)
np.save(sys.argv[5]+'_Normal_Input.npy',Dataset5_Input)
np.save(sys.argv[5]+'_Normal_Labels.npy',Dataset5_Label)

