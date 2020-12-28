import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def loader():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    return (train_images, train_labels), (test_images, test_labels)

def trainer():
    (train_images, train_labels), (test_images, test_labels) = loader()
           
    model = Sequential([layers.Flatten(input_shape=(28,28)), 
                        layers.Dense(512, activation='relu'), 
                        layers.Dense(256, activation='relu'), 
                        layers.Dense(128, activation='relu'), 
                        layers.Dense(10, activation='softmax') ])
            
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
    print('Test accuracy:', test_acc)
    
    predictions = model.predict(test_images)
    
    tester(test_images, predictions)
    
def tester(test_images, predictions):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    while True:
        i = int(input("Enter the number of image between 1 and 1000: "))
        plt.figure()
        plt.imshow(test_images[i], cmap="gray")
        plt.show()
        print(class_names[np.argmax(predictions[i])])
        c = input("Continue [y]/n: ")
        if c == 'n':
            break
        
trainer()