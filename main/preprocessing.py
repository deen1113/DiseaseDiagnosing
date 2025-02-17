import tensorflow as tf 
from tensorflow import ImageDataGenerator

'''
This is where data preprocessing occurs.
Using ImageDataGenerator, variations of the training images are made 
to help the model learn and generalize better.
Then, the images are resized to consistent shapes and delivered in batches to the model
using flow_from_directory.
Validation and testing images are also resized for consistency and use in the model.
'''

# training images are altered to help model learn variations
train_datagen = ImageDataGenerator(
    rescale= 1./255,            
    rotation_range= 20,         
    width_shift_range= 0.1,     
    height_shift_range= 0.1,    
    shear_range= 0.2,           
    zoom_range= 0.2,            
    horizontal_flip= True,      
    fill_mode= 'nearest'        
)

# each image is resized to work with model
train_generator = train_datagen.flow_from_directory(
    directory = "/processed_data/train",
    shuffle = True,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    seed = 42
)

# validation and testing images are resized to work with model as well
val_test_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_test_datagen.flow_from_directory(
    directory='processed_data/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)