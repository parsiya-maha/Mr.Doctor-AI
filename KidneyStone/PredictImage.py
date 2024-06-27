# Import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .LoadModel import load_model
import os
import shutil
import numpy as np

# Image size
__image_size = (150, 150)

# All class type
__CLASS_TYPES = ['Cyst', 'Normal', 'Stone', 'Tumor']


# Make image generator 
def __generate_image(test_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=__image_size,
        shuffle = False,
        class_mode='categorical',
        batch_size=1,
        seed=111
        )

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    return test_generator,nb_samples



# Function to make image ready to go in AI model
# => 
# cwd
# ---- test_dir
# ----     \   ----test2

def __add_prepare_image(image_path):
    """
    return test dir path
    """

    test_dir = "test_dir"
    os.mkdir(test_dir)
    s_test_dir = "test2"
    os.mkdir(os.path.join(test_dir,s_test_dir))

    main_path = os.path.split(image_path)[1]

    try :
        shutil.move(image_path,os.path.join(test_dir,s_test_dir,main_path))
    except :
        pass
    
    return test_dir



# Delete tree of folder -> [test_dir , test2]
def __delete_dir(dir_path,image_path):

    test_dir = "test_dir"
    s_test_dir = "test2"
    main_path = os.path.split(image_path)[1]
    shutil.move(os.path.join(test_dir,s_test_dir,main_path),image_path)
    
    shutil.rmtree(dir_path)




# Main function to predict image with [ image path , h5 path , json path ]
def predict_image(image_path,h5_path,json_path) :
    """
    return image type
    """
    # Prepare image 
    test_dir = __add_prepare_image(image_path)

    # Load model form LoadModel.py
    model = load_model(h5_path,json_path)
    test_generator,nb_samples = __generate_image(test_dir)

    # Predict image
    predict = model.predict_generator(test_generator,steps = nb_samples)

    # Delete dir
    __delete_dir(test_dir,image_path)

    # Find predict class name 
    predict_index = np.argmax(predict)

    return __CLASS_TYPES[predict_index]



#p = predict_image(r"D:\Parsia Works\python\Project\AI\BrainTumors\datasets\Testing\notumor\Te-no_0057.jpg",r"D:\Parsia Works\python\Project\AI\BrainTumors\model.h5",r"D:\Parsia Works\python\Project\AI\BrainTumors\model.json")
#print(p)