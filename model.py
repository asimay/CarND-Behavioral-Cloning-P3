import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import os

# read file resources related.
default_cur_dir = os.getcwd()
print('default_cur_dir is :', default_cur_dir)

lines = []
file_path = '.\\data\\driving_log.csv'
with open(file_path) as csvfile:
    reader = csv.reader(csvfile)
    for fileline in reader:
        lines.append(fileline)

lines.remove(lines[0])
print()
print('lines[0]:',lines[0])
print("length of data is:",len(lines))

# split the training and validation data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, valid_samples = train_test_split(lines, test_size=0.1)

print()
print("length of train_samples is:",len(train_samples))
print("length of valid_samples is: {}, percent={}".format(len(valid_samples), len(valid_samples)/len(lines)))

print()
print('Start to handling image...')

# process image, from BGR to RGB
def process_image(image_path):
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image
    
# process all image path
def process_files(file_lines):
    images = []
    measurements = []
    cur_dir = default_cur_dir + '\\data\IMG\\'
    #print('cur_dir is: ', cur_dir)
    
    for line in file_lines:
        #cur_dir = default_cur_dir + '\\data\IMG\\'
        center_source_path = line[0].strip()
        #print('line[0]:', line[0])
        center_filename = center_source_path.split('/')[-1]
        center_img_path = cur_dir + center_filename
        img_center = center_img_path
        
        left_source_path = line[1].strip()
        #print('line[1]:', line[1])
        left_filename = left_source_path.split('/')[-1]
        left_img_path = cur_dir + left_filename
        img_left = left_img_path
        
        right_source_path = line[2].strip()
        #print('line[2]:', line[2])
        right_filename = right_source_path.split('/')[-1]
        right_img_path = cur_dir + right_filename
        img_right = right_img_path


        #print('before length of images:', len(images))
        images.extend((np.asarray(img_center), np.asarray(img_left), np.asarray(img_right)))
        #print('after length of images:', len(images))

        #print('line[3]:', line[3])
        steer_center = float(line[3])
        #create adjusted steering measurements for the side camera images
        correction = 0.2
        steer_left = steer_center + correction
        steer_right = steer_center - correction
        
        measurements.extend((steer_center, steer_left, steer_right))
        
    images, measurements = shuffle(images, measurements)
    return (images, measurements)
        
X_train, y_train = process_files(train_samples)
X_valid, y_valid = process_files(valid_samples)
print('length of X_train:', len(X_train))
print('length of X_valid:', len(X_valid))

image_shape = (160, 320, 3)

# show steering angles histogram
def histogram(values, bins):
    plt.hist(values, bins=bins)
    plt.grid()
    plt.title("Histogram of unmodified steering angles")
    plt.show()
    
# data augmentation, for brightness
def augment_brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    augment = .25+np.random.uniform(low=0.0, high=1.0)
    img[:,:,2] = img[:,:,2]*augment
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img
          
# data augmentation flow, flip data.
def image_augmentation_flow(img, angle):
    img = augment_brightness(img)
    img = np.array(img)
    if np.random.randint(2):
        img = cv2.flip(img, 1)
        angle = -angle
    return img, angle

# generator, for batch handle the data
def generator(data, angle, batch_size=32):
    index = np.arange(len(data))
    batch_train = np.zeros((batch_size, 160, 320, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size), dtype = np.float32)
    print('data:')
    while 1:
       for i in range(batch_size):
            random = int(np.random.choice(index,1))
            img = cv2.imread(str(data[random]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = process_image(data[random])
            batch_train[i], batch_angle[i] = image_augmentation_flow(img, angle[random])
            
       yield (batch_train, batch_angle)

# Not used. generator, for batch handle the data
def generator2(data, angle, batch_size=32):
    #center, left, right
    correction_rate = [0.0, 0.3, -0.3]
    num_samples = len(data) 
    
    shuffle(data, angle)
    while 1: 
        # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            images = []
            measurements = []
            end = offset + batch_size
            batch = lines[offset:end]
            
            for line in batch:
                measurement = float(line[3])
                for i in range(3):
                    filename = line[i]
                    print('filename', filename)
                    if(len(filename.strip()) > 0):
                        img = cv2.imread(str(filename))
                        if(img is not None):
                            measurement = measurement + correction_rate[i]
                            #img = change_brightness(img)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                            images.append(img)
                            measurements.append(measurement)
                            img_flipped = np.fliplr(img)
                            images.append(img_flipped)
                            measurements.append(-measurement)		#flipped measurement
                            

            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield shuffle(X_train, y_train)	
       
print("Begin to train generator...")

#train_generator = generator2(X_train, y_train, batch_size=32)
#valid_generator = generator2(X_valid, y_valid, batch_size=32)

# create the NN module
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D  #Dropout ,Activation,
from keras.layers.convolutional import Conv2D
#from keras.layers.pooling import MaxPooling2D


print("Begin to train model...")

model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1, input_shape=image_shape, output_shape=image_shape))  #(x/255.0) - 0.5
model.add(Cropping2D(cropping=((70, 25),(0, 0))))

# pilotNet Network Module
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))

model.add(Flatten())
#model.add(Dense(1164))
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

BATCH_SIZE = 32
EPOCHS = 3

steps_per_epochss = int(len(X_train)/BATCH_SIZE)
validation_stepss = int(len(X_valid)/BATCH_SIZE)
print('steps_per_epochss is: ', steps_per_epochss)
print('validation_stepss is: ', validation_stepss)

# compile the module
model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])

print("Begin to fit generator...")

# fit generator
history_object = model.fit_generator(generator(X_train, y_train, batch_size=32), steps_per_epoch=steps_per_epochss, epochs=EPOCHS, verbose=1,
                                     validation_data=generator(X_valid, y_valid, batch_size=32), validation_steps=validation_stepss )   #nb_epoch=5

### print the keys contained in the history object
print(history_object.history.keys())

print("Save as model.h5...")
model.save('model.h5')
print("Done...")

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#print("Testing")

# Evaluate the test data in Keras Here
#metrics = model.evaluate(X_normalized_test, y_one_hot_test)
# TODO: UNCOMMENT CODE
#for metric_i in range(len(model.metrics_names)):
#    metric_name = model.metrics_names[metric_i]
#    metric_value = metrics[metric_i]
#    print('{}: {}'.format(metric_name, metric_value))
    
# show image
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
#ax1.imshow(images[1])
#ax1.set_title('Original Image', fontsize=10)
#ax2.imshow(images[2])
#ax2.set_title('RGB Image', fontsize=10)

#print('images shape:', str(images[1].shape))
#print('images length:', len(images))

print("OK")

