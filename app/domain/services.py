from app.domain.models import Message

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import StringLookup
from tensorflow import keras
import cv2
import tempfile
import os

max_len = 8
image_width = 128
image_height = 64+40
img_size=(image_width, image_height)

characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

prediction_model = tf.keras.models.load_model('app/api/v1/my_model5')

def get_message() -> str:
    message = Message(content="Hello, FastAPI with DDD and Poetry!")
    return message.content

def preprocess_image(image_path, img_size=(image_width, image_height)):
    

    image = image_path
    # image = tf.io.read_file(image_path)
    # image = tf.image.decode_png(image, 1)

    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
        constant_values=255,
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)

    image = tf.cast(image, tf.float32) / 255.0

    return image

def decode_prediction(pred):
    results = keras.backend.ctc_decode(
        pred,
        input_length=[pred.shape[1]],
        greedy=True
    )[0][0][:, :max_len]

    text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        text.append(res)
    return text

def cropImg(path):
  # Read image from which text needs to be extracted
#   with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
#         temp_filename = temp_file.name
#         temp_file.write(path)

  img = cv2.imread(path)
  # Preprocessing the image starts

  # Convert the image to gray scale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Performing OTSU threshold
  ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

  # Specify structure shape and kernel size.
  # Kernel size increases or decreases the area
  # of the rectangle to be detected.
  # A smaller value like (10, 10) will detect
  # each word instead of a sentence.
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

  # Applying dilation on the threshold image
  dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

  # Finding contours
  contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_NONE)

  # Creating a copy of image
  im2 = img.copy()

  # A text file is created and flushed
  file = open("recognized.txt", "w+")
  file.write("")
  file.close()

  # Looping through the identified contours
  # Then rectangular part is cropped and passed on
  # to pytesseract for extracting text from it
  # Extracted text is then written into the text file
  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Create a rectangle on a blank image
    # rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Cropping the text block for visualization
    cropped = im2[y : y + h, x : x + w]

    add_height = max(0, 128 - cropped.shape[0])

    # Add height on top of image. Fill with color (255,255,255), that is white
    img_final = cv2.copyMakeBorder(cropped, add_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # os.remove(temp_filename)

    # img_final = cv2.imencode('.png', img_final)[1].tobytes()

    return img_final

def IA(image_bytes):
    # img = preprocess_image('/Users/hmsb/Desktop/dtsetIA/API/J.png')
    # print(image_bytes)

    # try:        

        img_final = cropImg('ar.png')
        # Converte a imagem para uma string PNG
        image_encoded = tf.io.encode_png(img_final)

        # Decodifica a imagem PNG
        image_decoded = tf.image.decode_png(image_encoded, 1)

        # image_decoded = tf.image.decode_png(tf.convert_to_tensor(cropImg('a.png')), 1)


        # image_decoded = tf.image.decode_png(cropImg('a.png'), 1)
        
        img = preprocess_image(image_decoded)

        # # Make Prediction
        pred = prediction_model.predict(np.array([img]))

        # Decode Prediction
        output_text = decode_prediction(pred)

        return output_text

    # except Exception as e:

    #     print(f'Erro:{e}')

    #     image_decoded = tf.image.decode_png(image_bytes, 1)

    #     img = preprocess_image(image_decoded)

    #     # Make Prediction
    #     pred = prediction_model.predict(np.array([img]))

    #     # Decode Prediction
    #     output_text = decode_prediction(pred)
    #     return 'Não foi'