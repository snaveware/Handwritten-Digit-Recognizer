import os
from sys import breakpointhook
from sys import exit
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print(" IT GROUP 2 PRESENTS ... \n")
print(" THE HANDWRITTEN DIGITS RECOGNIZER \n")



# Shows options to the user

        
def showOptions(): 
    # decision variables Loads default model by default
    global trainNewModel 
    global loadRecentModel 
    global loadDefaultModel 

    # Show options
    exit = False
    while not exit:
        options = """

        1. Load default model to predict data under ./digits
        2. Load the Most recent model to predict data under ./digits
        3. Create and Train a new Model
        4. Exit

        """

        print(options)

        # Take the chosen option
        choice = input("Choice: ")

        # Decide the action
        print('choice', choice)

        try: 
            choice = int(choice)
            if choice == 2: 
                loadRecentModel = True
                loadDefaultModel = False
                trainNewModel = False
                print('Loading recent model...')
                break
            elif choice == 3:
                loadRecentModel = False
                loadDefaultModel = False 
                trainNewModel = True  
                print('Training new Model...')  
                break
            elif choice == 1: 
                print("Loading Default Model")
                trainNewModel = False
                loadRecentModel = False
                loadDefaultModel = True
                break
            elif choice == 4:
                print("Goodbye😎")
                exit = True
                break
            else: 
                print('😂 Stop being naughty, I need a valid option')
                
        except: 
            print("\n 🤭 oops! Error! I'm Dead! \n" )  

showOptions()            

# Loads and predicts images from ./digits
def predictCustomImages(model): 

    # Load custom images from /digits and predict them
    imageNumber = 1
    while os.path.isfile('digits/digit{}.png'.format(imageNumber)):
        try:
            img = cv2.imread('digits/digit{}.png'.format(imageNumber))[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"The number is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
            imageNumber += 1
        except:
            print(f"Error reading image {imageNumber}  Proceeding with next image...")
            imageNumber += 1



if trainNewModel:
    # Loading the MNIST data set from Keras and splitting the data set into train and test data
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=3)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Saving the model
    model.save('Recent_Handwritten_Digits_Recognizer.model')
    showOptions()

elif  loadRecentModel:
    # Load the most Recently created model
    model = tf.keras.models.load_model('Recent_Handwritten_Digits_Recognizer.model')
    predictCustomImages(model)
elif loadDefaultModel:
    # Load the Default Model
    model = tf.keras.models.load_model('Handwritten_Digits_Recognizer.model')
    predictCustomImages(model)


