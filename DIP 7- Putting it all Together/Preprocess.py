import cv2
import numpy as np

def get_kernels(kernel_size = (3,3)):
    """
    Get kernel for thinning
    """

    K1 = np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]])
    K2 = np.array([[0, -1, -1], [1, 1, -1], [0, 1, 0]])

    kernels = []
    kernels.append((K1, K2))

    def rotate3x3(kernel):
        kernel_temp = np.empty_like(kernel)

        kernel_temp[0, :] = kernel[::-1, 0]
        kernel_temp[:, -1] = kernel[0, :]
        kernel_temp[-1, :] = kernel[::-1, -1]
        kernel_temp[:, 0] = kernel[-1, :]

        kernel_temp[1,1] = kernel[1,1]

        return kernel_temp

    for _ in range(3):
        K1, K2 = rotate3x3(K1), rotate3x3(K2)
        kernels.append((K1, K2))
    
    return kernels
        

def preprocess(image):
    """
    image : image path or np.array of the binary image

    returns a preprocessed image for the model.
    """
    
    if isinstance(image, str):
        im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        im = image * 255

    im = cv2.resize(im, (45, 45))
    im = (im<127).astype("uint8")

    kernels = get_kernels()

    thinned_im = im * 255 #thinning image
    prev_thinned_im = np.copy(thinned_im)
    while True:
        for k1, k2 in kernels:
            h_k1 = cv2.morphologyEx(thinned_im.astype("uint8"), cv2.MORPH_HITMISS, k1)
            thinned_im = np.logical_and(thinned_im.astype("bool"), np.logical_not(h_k1.astype("bool"))) * 255

            h_k2 = cv2.morphologyEx(thinned_im.astype("uint8"), cv2.MORPH_HITMISS, k2)
            thinned_im = np.logical_and(thinned_im.astype("bool"), np.logical_not(h_k2.astype("bool"))) * 255
        
        prev_thinned_im = np.copy(thinned_im)
        if np.any(thinned_im == prev_thinned_im):
            break
    
    return thinned_im

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.imshow(preprocess("./Expression/4.jpg"))
    plt.show()

