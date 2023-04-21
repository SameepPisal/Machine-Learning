import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Directory where the original images are stored
original_dir = 'C:/Users/Administrator/Downloads/archive/dataset/train/good'

# Directory where the augmented images will be saved
augmented_dir = 'C:/Users/Administrator/Downloads/archive/dataset/marble/train/good'

# Create an instance of the ImageDataGenerator class with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Loop through all the images in the original directory and apply augmentation to each image
for filename in os.listdir(original_dir):
    # Load the image
    img = load_img(os.path.join(original_dir, filename))

    # Convert the image to a numpy array
    x = img_to_array(img)

    # Reshape the input to (1, img_width, img_height, 3) to match the model's input shape
    x = x.reshape((1,) + x.shape)

    # Use the datagen instance to generate new augmented images
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir, save_prefix=filename[:-4], save_format='jpg'):
        i += 1
        if i > 9: # Generate 5 augmented images for each original image
            break
