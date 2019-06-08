import cv2
import dlib
import numpy as np
from faceswap import  FaceSwap
import matplotlib.pyplot as plt



def main():
    im1 = cv2.imread('images/img1.JPG')
    im2 = cv2.imread('images/img2.JPG')
    img = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    print(img)
    plt.imshow(img)
    plt.show()


    fs = FaceSwap()

    # make fake
    img_swapped = fs.astype(np.uint8).swap_faces(im1, im2)
    img_swapped = cv2.cvtColor(img_swapped, cv2.COLOR_BGR2RGB)
    print("max=", np.max(img_swapped))
    print("min=", np.min(img_swapped))

    plt.imshow(img_swapped)
    plt.show()



if __name__=="__main__":
    main()