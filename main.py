from __future__ import absolute_import, division, print_function, unicode_literals

from globals import * # Import globals.py like this in every file you need to access the global variables/use the libraries imported there

# https://www.tensorflow.org/tutorials/keras/

print("Installed Tensflow version: {}".format(tf.__version__))

fashion_mnist = keras.datasets.fashion_mnist

# This will load the fashion_mnist data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names. This is what we categorize each clothing item into.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale the images down to the correct color scale, so we scale them from (0-255) to (0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Setup the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # This turns the 28x28 pixel images into a 1d array of 28*28=784 pixels
    keras.layers.Dense(128, activation=tf.nn.relu), #A dense node layer with 128 nodes
    keras.layers.Dense(10, activation=tf.nn.softmax) #A dense node layer with 10 nodes, one for each class
])
#Dense layers are densely-connected, aka fully-connected
#The softmax layer returns an array of 10 probability scores that sum to 1
#Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes

# Compile the model!
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
#Loss function: Measures how accurate the model is during training.
#Tutorial says: 'We want to minimize this function to "steer" the model in the right direction'. Not sure yet what this exactly means.
#According to my friend it means we try to keep the function as simple possible. Still not 100% sure on this, but I did find some more info.
#You basically use a form of categorical crossentropy if your end result sorts into specifics classes.
#If your categories are just integers, then you use sparse_categorical_crossentropy
#If your categories are "one-hot encodings" (ex: [1,0,0]), then you use categorical_crossentropy
#I will have to dive into this more, it seems pretty important

# Tell the model to train using 'train_images' as input with 'train_labels' as the correct output
model.fit(train_images, train_labels, epochs=5) #I think epochs just means the amount of times it's gonna go through the training data. Confusing name

# Evaluating the test images using our model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: {}".format(test_acc))
#The accuracy on the test dataset is lower than the accuracy on the training dataset. This is called 'overfitting'
#Overfitting is where a ML model performs worse on new data than on the training data
