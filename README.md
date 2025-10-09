Code for the dataset, to use in python.

The python_data.py code, transforms the data from .mat files into a csv file that can be read with pandas library, the result .csv is called output.csv.
The load_images file adds the image files to the pandas, this code adds the image path to each row in the dataset for better usage. 
The get_vi_images.py, is used to generate the vegetation indices and the descriptors for each vi image, you can save the vi image as .csv but it will need a lot of space in disk. The end result is the Data_with_descriptors.csv which is a
Dataframe that contains all the vegetation indices from spectral reflectance values and from the images. 

The utils.py contains important functions such as the get_spectral that receives as input the dataframe and returns the spectral values from 350 to 25000 nm as an array. 
To load the .tiff files the cv2.imread function is recommended using the structure cv.imread(img_path, -1), if the -1 is not added in the function it will not properly read the 16 bit function, the same goes for saving any image, 
if cv.imwrite(impath
