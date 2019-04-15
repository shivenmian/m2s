import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from l0smooth.L0_serial import l0_smooth

'''
img = cv2.imread('img2.png')
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

edges = cv2.Canny(image_l0.astype('uint8'), 100, 200)

plt.show()
'''
def main():

    parser = argparse.ArgumentParser(description="mural2sketch")

    parser.add_argument('image', help="input image file")

    args = parser.parse_args()

    image = args.image

    # Preprocessing

    image_l0 = l0_smooth(image)

    image_blur = cv2.GaussianBlur(image_l0, (5, 5), 0)

    # Outer Edge Detection

    image_can = cv2.Canny(image_blur.astype('uint8'), 100, 200)

if __name__ == '__main__':
    main()