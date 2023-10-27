from app.domain.models import Message

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import StringLookup
from tensorflow import keras

from app.domain.services3 import cropImg

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

def IA(image_bytes):
    # img = preprocess_image('/Users/hmsb/Desktop/dtsetIA/API/J.png')
    # print(image_bytes)

    image_bytes = cropImg(image_bytes)

    image_decoded = tf.image.decode_png(image_bytes, 1)

    img = preprocess_image(image_decoded)

    # Make Prediction
    pred = prediction_model.predict(np.array([img]))

    # Decode Prediction
    output_text = decode_prediction(pred)
    return output_text
