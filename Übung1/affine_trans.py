import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage.color import rgb2gray


def rotation_matrix(alpha):
    a_rad = np.radians(alpha)
    rot_matrix = np.array([[np.cos(a_rad), - np.sin(a_rad)],[np.sin(a_rad), np.cos(a_rad)]])
    return rot_matrix

def get_empty_plane(image, a, xl):
    if xl:
        factor = 5
    else:
        factor = 2

    diagonal = int(np.ceil(np.sqrt(np.square(image.shape[0]) + np.square(image.shape[1]))))
    max_width = factor * diagonal 
    trans_image = np.empty([max_width + 2*np.abs(a[0]), max_width + 2*np.abs(a[1]), 3])
    return trans_image

def floor(num):
    return int(np.floor(num))

def ceil(num):
    return int(np.ceil(num))

def affine_transform(image, matrix, a=np.array([0,0]), interpolate=False, xl=False):
    trans_image = get_empty_plane(image, a, xl)
    
    # mid of the empty plane
    t_mid_x = round(trans_image.shape[0]/2)
    t_mid_y = round(trans_image.shape[1]/2)

    A_inv = np.linalg.inv(matrix)

    for i in range(trans_image.shape[0]):
        for j in range(trans_image.shape[1]):
            x_trans = np.array([i,j])
            x_og = A_inv.dot(x_trans - a)

            if interpolate:
                if 0 <= round(x_og[0]) < image.shape[0]-1 and 0 <= round(x_og[1]) < image.shape[1]-1:
                    # little hack to avoid black stripes that occur when a pixl-coordinate is an integer
                    if (x_og[0].is_integer and x_og[0] < image.shape[0]):
                        x_og[0] += 0.0000001
                    if (x_og[1].is_integer and x_og[1] < image.shape[1]):
                        x_og[1] += 0.0000001

                    p1 = image [floor(x_og[0])] [floor(x_og[1])]
                    p2 = image [floor(x_og[0])] [ceil(x_og[1])]
                    p3 = image [ceil(x_og[0])] [floor(x_og[1])]
                    p4 = image [ceil(x_og[0])] [ceil(x_og[1])]

                    a1 = (x_og[0] - np.floor(x_og[0])) * (x_og[1] - np.floor(x_og[1]))
                    a2 = (x_og[0] - np.floor(x_og[0])) * (np.ceil(x_og[1]) - x_og[1])
                    a3 = (np.ceil(x_og[0]) - x_og[0]) * (x_og[1] - np.floor(x_og[1]))
                    a4 = (np.ceil(x_og[0]) - x_og[0]) * (np.ceil(x_og[1]) - x_og[1])
        
                    gx = a4*p1 + a3*p2 + a2*p3 + a1*p4

                    trans_image[t_mid_x + i][t_mid_y + j] = gx

            elif not interpolate:
                if 0 <= round(x_og[0]) < image.shape[0] and 0 <= round(x_og[1]) < image.shape[1]:
                    gx = image[round(x_og[0])][round(x_og[1])]
                    trans_image[t_mid_x + i][t_mid_y + j] = gx

    result_image = trans_image[t_mid_x : t_mid_x + image.shape[0], t_mid_y : t_mid_y + image.shape[1]]
    return result_image

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":
    # Load Image
    image = skimage.io.imread(fname="./Übung1/gletscher.jpg")
    #image = skimage.io.imread(fname="./ambassadors.jpg")

    image = rgb2gray(image) # already normalized
    # image = image.astype('float64')

    # Normalize Image
    #image = normalize(image)
    # rot 45 skal rot -45

    A1 = rotation_matrix(45)
    A2 = np.array([[1.5 ,0],[0, 0.5]])
    A3 = rotation_matrix(-45)

    A = A1 @ A2 @ A3


    #A = np.array([[2 ,0],[2, 2]])
    #A = np.array([[1 ,0],[1.5, 1]])

    a = np.array([0, 0])
    
    trans_image = affine_transform(image, A, a, interpolate=False, xl=True)

    fig = plt.figure(figsize=(1, 2))
    fig.add_subplot(1, 2, 1)
    plt.imshow((image*255).astype(int), cmap='gray')

    fig.add_subplot(1, 2, 2)
    plt.imshow((trans_image*255).astype(int), cmap='gray')

    plt.show()
