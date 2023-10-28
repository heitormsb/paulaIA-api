import cv2
import pytesseract
import matplotlib.pyplot as plt
import os
import numpy as np

def cropImg(path):
  # Read image from which text needs to be extracted
  # img = cv2.imread("/content/by_field/hsf_0/upper/49/49_00001.png")

  # if aa == 'a':
    # img = cv2.imread(path)
  # else:
  nparr = np.frombuffer(path, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 

  # Convert the image to gray scale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Performing OTSU threshold
  ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

  # Specify structure shape and kernel size.
  # Kernel size increases or decreases the area
  # of the rectangle to be detected.
  # A smaller value like (10, 10) will detect
  # each word instead of a sentence.
  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))

  # Applying dilation on the threshold image
  dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

  # Finding contours
  contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_NONE)

  # Creating a copy of image
  im2 = img.copy()

  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Create a rectangle on a blank image
    # rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # print(w) 
    # if w < 25:
    #   continue

    # Cropping the text block for visualization
    cropped = im2[y : y + h, x : x + w]

    # if height is > 128, resize to 128
    if cropped.shape[0] >= 128:
      cropped = cv2.resize(cropped, (0,0), fx=80/cropped.shape[0], fy=80/cropped.shape[0])
    # cropped = cv2.resize(cropped, (0,0), fx=1/4, fy=1/4)
    
    add_height = max(0, 128 - cropped.shape[0])

    # Add height on top of image. Fill with color (255,255,255), that is white
    padded = cv2.copyMakeBorder(cropped, add_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # if aa == 'a':
        # cv2.imwrite(f'/home/hmsb/paulaIA-api/PLIM_TESTE.png', padded)

    retval, buffer = cv2.imencode('.png', padded)
    binary_image = buffer.tobytes()
    return binary_image


# cropImg("/home/hmsb/paulaIA-api/letra/p1.png", aa='a')