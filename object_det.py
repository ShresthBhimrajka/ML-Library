from tensorflow.keras import datasets
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def loader():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0 
    return (train_images, train_labels), (test_images, test_labels)

def trainer():
    (train_images, train_labels), (test_images, test_labels) = loader()
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    
        
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10))
        
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    
    predictions = model.predict(test_images)
    
    tester(test_images, predictions)
    
def tester(test_images, predictions):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    while True:
        i = int(input("Enter the number of image between 1 and 10000: "))
        plt.figure()
        plt.imshow(test_images[i], cmap="gray")
        plt.show()
        print(class_names[np.argmax(predictions[i])])
        c = input("Continue [y]/n: ")
        if c == 'n':
            break

trainer()