# Face_Recognition_Human_Vision

## UpdatedEccentricityCNN.py 
To get the top 5 test accuracy for the dataset specified using Eccentricity CNN based network.Run the code using command python3 UpdatedEccentricityCNN.py(specify the GPU before python3).Before running the code mention the train and test dataset directory in the code as given
- dataSet_directory = 'train_data'
- testData_directory2 ='test_data1'
- testData_directory3 ='test_data2'
- testData_directory4 = 'test_data3'
- testData_directory5 = 'test_data4'                             
  
"dataSet_directory" contains the directory on which the model is trained leaving apart 20% as test data

"testData_directory2","testData_directory3","testData_directory4","testData_directory5" contains differently scaled and shifted  version of the faces on which the trained model is tested.

The final result is the top 5 test accuracy for each of the different version of faces.
