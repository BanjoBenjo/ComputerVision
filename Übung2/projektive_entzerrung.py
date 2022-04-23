import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage.color import rgb2gray

def projective_qualization_parameter(object_points, picture_points):
    # xi ist Objektkoordiante
    # xi_ ist die Bildkoordinate
    x1 = object_points[0][0]
    y1 = object_points[0][1]
    x2 = object_points[1][0]
    y2 = object_points[1][1]
    x3 = object_points[2][0]
    y3 = object_points[2][1]
    x4 = object_points[3][0]
    y4 = object_points[3][1]

    x_1 = picture_points[0][0]
    y_1 = picture_points[0][1]
    x_2 = picture_points[1][0]
    y_2 = picture_points[1][1]
    x_3 = picture_points[2][0]
    y_3 = picture_points[2][1]
    x_4 = picture_points[3][0]
    y_4 = picture_points[3][1]

    M = np.array([
        [x_1, y_1, 1, 0, 0, 0, -x1*x_1, -x1*y_1],
        [0, 0, 0, x_1, y_1, 1, -y1*x_1, -y1*y_1],
        [x_2, y_2, 1, 0, 0, 0, -x2*x_2, -x2*y_2],
        [0, 0, 0, x_2, y_2, 1, -y2*x_2, -y2*y_2],
        [x_3, y_3, 1, 0, 0, 0, -x3*x_3, -x3*y_3],
        [0, 0, 0, x_3, y_3, 1, -y3*x_3, -y3*y_3],
        [x_4, y_4, 1, 0, 0, 0, -x4*x_4, -x4*y_4],
        [0, 0, 0, x_4, y_4, 1, -y4*x_4, -y4*y_4]
    ])

    x_vec = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
    M_inv = np.linalg.inv(M)

    a_vec =  M_inv @ x_vec

    return a_vec


def projektive_equalization(image, a_vec):
    equal_image = np.empty(image.shape)

    a1 = a_vec[0]
    a2 = a_vec[1]
    a3 = a_vec[2]
    b1 = a_vec[3]
    b2 = a_vec[4]
    b3 = a_vec[5]
    c1 = a_vec[6]
    c2 = a_vec[7]
    
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_value = image[x][y]
            
            x_ = (( (b2 - c2*b3)*x + (a3*c2 - a2)*y + a2*b3 - a3*b3 ) /
                ( (b1*c2 - b2*c1)*x + (a2*c1 - a1*c2)*y + a1*b2 - a2*b1 ))

            y_ = (( (b3*c1 - b1)*x + (a1 - a3*c1)*y + a3*b1 - a1*b3 ) /
                ( (b1*c2 - b2*c1)*x + (a2*c1 - a1*c2)*y + a1*b2 - a2*b1 )) 

            print(x_)
            print(y_)

            equal_image[x_][y_] = pixel_value

    return equal_image

if __name__ == "__main__":
    # Load Image
    image = skimage.io.imread(fname="./Übung1/gletscher.jpg")
    #image = skimage.io.imread(fname="./ambassadors.jpg")

    image = rgb2gray(image) # already normalized

