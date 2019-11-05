from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import cv2
from random import shuffle

# -------------------- Configurations -------------------- #
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BATCH_SIZE = 32
NUM_CLASSES = 2
NUM_EPOCHS = 20
TRAIN_DIR = "./data/train"
TEST_DIR = "./data/test"

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keylime_trained_model.h5'


train_images = []
test_images = []
for filename in os.listdir(TRAIN_DIR):
    if 'png' in filename:
        train_images.append(TRAIN_DIR + "/" + filename) 
for filename in os.listdir(TEST_DIR):
    if 'png' in filename:
        test_images.append(TEST_DIR + "/" + filename)
shuffle(train_images)
shuffle(test_images)
print("Training samples: " + str(len(train_images)))
print("Testing samples: " + str(len(test_images)))

def train_generator():
    while True:
        for start in range(0, len(train_images), BATCH_SIZE):
            x_batch = []
            y_batch = []
            end = min(start + BATCH_SIZE, len(train_images))
            for img_path in range(start, end):
                img = cv2.imread(train_images[img_path])
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                x_batch.append(img)
                y_batch.append(['1']) if 'cat' in train_images[img_path] else y_batch.append(['0'])
            yield (np.array(x_batch), keras.utils.to_categorical(np.array(y_batch), NUM_CLASSES))
                                        # Note: Use keras.utils.np_utils.to)categorical to 
                                        # convert labels to categorical one-hot vectors

def test_generator():
    while True:
        for start in range(0, len(test_images), BATCH_SIZE):
            x_batch = []
            y_batch = []
            end = min(start + BATCH_SIZE, len(test_images))
            for img_path in range(start, end):
                img = cv2.imread(test_images[img_path])
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                x_batch.append(img)
                y_batch.append(['1']) if 'cat' in test_images[img_path] else y_batch.append(['0'])
            yield (np.array(x_batch), keras.utils.to_categorical(np.array(y_batch), NUM_CLASSES))





# ---------------------- Make and compile the model -------------------- #
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(
    train_generator(),
    epochs = NUM_EPOCHS,
    steps_per_epoch= len(train_images) // BATCH_SIZE,
    validation_data= test_generator(),
    validation_steps = len(test_images) // BATCH_SIZE,
)







# -------------------- OLD CODE -------------------- #
'''
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = True)

# testing if converting the big ass nested list to numpy array would help
x_train = numpy.array(x_train)
x_test = numpy.array(x_test)
y_train = numpy.array(y_train)
y_test = numpy.array(y_test)
print("finished reading input data... printing x_train to see the structure... ")
# print(X)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
# y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
'''

'''
if True:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size = BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(x_test, y_test),
              shuffle=True)
'''

'''
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
'''

'''
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
'''