# Face_Recognition_Human_Vision

## preprocessing.py
To create the input data for the eccentricity model. Run the code using command python3 preprocessing.py 'train_data' 'test_data1' 'test_data2' 'test_data3' 'test_data4'. Where the arguments after the script names are the names of the directory containing the training data followed by four different test data directory name. "train_data" contains the directory on which the model is trained leaving apart 20% as test data."test_data1","test_data2","test_data3","test_data4" contains differently scaled and shifted  version of the faces on which the trained model is tested. The output of the code will be the processed image and their corresponding labels in the form of .npy files for each of the train and test data for each of the given directories. Before running UpdatedEccentricityCNN.py , preprocessing.py needs to be run if .npy files for the train and test dataset does not exist.

## UpdatedEccentricityCNN.py 
To get the top 5 test accuracy for the dataset specified using Eccentricity CNN based network.Run the code using command (specify the GPU) python3 UpdatedEccentricityCNN.py 'train_data' 'test_data1' 'test_data2' 'test_data3' 'test_data4'. Where the arguments after the script names are the names of the directory containing the training data followed by four different test data directory name. "train_data" contains the directory on which the model is trained leaving apart 20% as test data."test_data1","test_data2","test_data3","test_data4" contains differently scaled and shifted  version of the faces on which the trained model is tested. Run preprocessing.py if .npy for the corresponding directory does not exist.

The final result is the top 5 test accuracy for each of the different version of faces.

## face_segmentation.py
To get the binary mask of the face and the background as well as the segmented face from the background. Run the code using command python3 face_segmentation.py. Before running the code specify the name of the directory containing the original cropped faces of each actor in the code of the variable facedir as given
- facedir = 'Cropped_Face'

The code will create two new directories containing the binary mask as well as the segmented mask for each actor.
For Segmentation an extra file needs to be imported "FCN8s_keras" which contains the pretrained network for segmenting the faces.The pretrained weights can be downloaded from this [link](https://drive.google.com/ucid=1alyR6uv4CHt1WhykiQIiK5MZir7HSOUU&export=download) 

## scaled_faces_green_background.py
To get 32,64,128,200 sized faces on a green background using the ouput of face_segmentation.py. Run the code using command python3 scaled_faces_green_background.py. The output of the code will be four new directories containing the scaled faces of each actor."green.jpg" image is used as background.

## scaled_faces_places365_background.py
To get 32,64,128,200 sized faces on places365 dataset using the ouput of face_segmentation.py. Run the code using command python3 scaled_faces_places365_background.py. The output of the code will be four new directories containing the scaled faces of each actor. The variable "background" in the code is the name of the directory conating the places 365 dataset. Each of the image in this directory is named in a numerical order(Eg. 0.jpg) so that same background image does not go in both test and train set which may affect the result.

## center_shifted_faces.py
To get 16 pixels left,right,up and down center shifted faces on green background.Run the code using command python3 center_shifted_faces.py. The output of the code will be four new directories containing the shifted faces of each actor."green.jpg" image is used as background.If instead of 16 pixels the shifting needs to be increased or decreased then changes need to be made only in the variabels "left_border","right_border","up_border","down_border" accordingly(eg if 8 pixels shift required change the value of shift variable to 8 and the faces will be shifted 8 pixels)
