import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
#import tensorflow as tf
#from tensorflow.keras.preprocessing import image

def process(image):   # Loading the image
    imagem_cinza = image
#    imagem_cinza = cv2.imread(caminho_imagem_png, cv2.IMREAD_GRAYSCALE)

    # Applying a blur to reduce noise
    imagem_cinza = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

    # Using edge detection to highlight letters
    bordas = cv2.Canny(imagem_cinza, 50, 150, apertureSize=3)
    contornos, _ = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting contours by horizontal position (left to right)
    contornos = sorted(contornos, key=lambda x: cv2.boundingRect(x)[0])

    letters_images = []
    converted_images = []

    # Extracting every letter
    for i, contorno in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(contorno)

        if w > 20 and h > 20:
            letter = imagem_cinza[y:y+h, x:x+w]
            letters_images.append(letter)

    # Display the letter images in order
    for i, letter in enumerate(letters_images):
        # plt.subplot(1, len(letters_images), i + 1)
        # plt.imshow(letter, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # Convert to EMNIST format (28x28 pixels, inverted colors)
        letter_resized = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_LINEAR)
        inverted_letter = cv2.bitwise_not(letter_resized)

        converted_images.append(inverted_letter)

#    print("Conversion to EMNIST format complete!")

    return converted_images

def IA2(image_bytes):
    #to transform de image_byte to a image file
    nparr = np.fromstring(image_bytes, np.uint8)
    image_decoded = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    #image_decoded = tf.image.decode_png(image_bytes, 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        converted_images = process(image_decoded)
        modelo_carregado = joblib.load('/Users/henriquenino/Desktop/PEBIX-PAULA/aprendizado/mlp3')
        typed_story = ""
        for letter in converted_images:
            single_item_array = (np.array(letter)).reshape(1,784)
            prediction = modelo_carregado.predict(single_item_array)
            typed_story += str(chr(prediction[0]+64))
#           print("Conversion to typed story complete!")
    #       print(typed_story)
        return typed_story
#
#def IA2(image_bytes):
#    image_decoded = tf.image.decode_png(image_bytes, 1)
#    with warnings.catch_warnings():
#        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
#        converted_images = process(image_decoded)
#        modelo_carregado = joblib.load('/Users/henriquenino/Desktop/PEBIX-PAULA/aprendizado/mlp3')
#        typed_story = ""
#        for letter in converted_images:
#            single_item_array = (np.array(letter)).reshape(1,784)
#            prediction = modelo_carregado.predict(single_item_array)
#            typed_story += str(chr(prediction[0]+64))
##           print("Conversion to typed story complete!")
#    #       print(typed_story)
#        return typed_story
#
#def IA2(image_bytes):
#    with warnings.catch_warnings():
#        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
#        converted_images = process(image_bytes)
#        modelo_carregado = joblib.load('/Users/henriquenino/Desktop/PEBIX-PAULA/aprendizado/mlp3')
#        typed_story = ""
#        for letter in converted_images:
#            single_item_array = (np.array(letter)).reshape(1,784)
#            prediction = modelo_carregado.predict(single_item_array)
#            typed_story += str(chr(prediction[0]+64))
#
##        print("Conversion to typed story complete!")
#        return typed_story

#oi = IA2('/Users/henriquenino/Desktop/PEBIX-PAULA/aprendizado/a01-AARAO3.png')
#print(oi)