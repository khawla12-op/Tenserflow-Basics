import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
#Keras is an api
#loading data:
(x_train, y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(-1,28*28).astype("float32")/255.0
x_test=x_test.reshape(-1,28*28).astype("float32")/255.0
#How to create a sequential API:
model=keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512,activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)
#if i want want to see all the layers:
features=model.predict(x_train)
for feature in features:
    print(feature.shape)
#to print the summary
print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)
#How to create a functional API:
inputs=keras.Input(shape=(784))
x=layers.Dense(512,activation='relu',name='first_layer')(inputs)
x=layers.Dense(256, activation='relu',name='second_layer')(x)
outputs=layers.Dense(10,activation='softmax')(x)
model=keras.Model(inputs=inputs,outputs=outputs)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)
