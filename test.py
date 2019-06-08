import cv2
import dlib
import numpy
import faceswap
import matplotlib.pyplot as plt



def main():
    im1 = cv2.imread('images/img1.JPG')
    im2 = cv2.imread('images/img1.JPG')
    plt.imshow(im1.transpose())
    plt.show()


    # make fake
    img_swapped = faceswap.swap_faces(im1, im2)
    plt.imshow(img_swapped)
    plt.show()



if __name__=="__main__":
    main()