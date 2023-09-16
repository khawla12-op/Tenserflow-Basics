import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
#Loads data into 4 variables:x_train,y_train are the training images while x_test,y_test are the test images
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
#Normalize the pixels values of the images to the range [0,1]
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

model=keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),  #32 for height ,32 for width and 3 channels for rgb
        layers.Conv2D(32,3,padding='valid',activation ='relu'),#valid will change depending on the kernel size
        layers.MaxPooling2D(pool_size=(2,2)),


    ]
)


    model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"]
)
model.fit(x_train ,y_train , batch_size=64,epochs=10,verbos=2)
model.evaluate(x_test ,y_test,batch_size=64,verbos=2)
#Training with the functional api:
#This function define the layers and architecture of my neural network model
def my_model():
    # Define the input layer with a shape of (32, 32, 3)
    inputs = keras.Input(shape=(32, 32, 3))

    # Apply a convolutional layer with 32 filters and a 3x3 kernel
    x = layers.Conv2D(32, 3)(inputs)

    # Apply batch normalization to the output of the convolutional layer
    x = layers.BatchNormalization()(x)

    # Apply the ReLU (Rectified Linear Unit) activation function
    x = keras.activations.relu(x)

    # Apply max pooling to reduce the spatial dimensions
    x = layers.MaxPooling2D()(x)

    # Apply another convolutional layer with 128 filters and a 3x3 kernel
    x = layers.Conv2D(128, 3)(x)

    # Apply batch normalization to the output of the second convolutional layer
    x = layers.BatchNormalization()(x)

    # Apply the ReLU activation function
    x = keras.activations.relu(x)

    # Flatten the output to prepare it for fully connected layers
    x = layers.Flatten()(x)

    # Apply a fully connected (dense) layer with 64 units and ReLU activation
    x = layers.Dense(64, activation='relu')(x)

    # Apply another fully connected layer without activation for the output
    outputs = layers.Dense(10)(x)  # Assuming there are 10 classes

    # Create the Keras Model with defined inputs and outputs
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create an instance of the model
model = my_model()

