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

    x_vec = np.transpose(np.array([[x1, y1, x2, y2, x3, y3, x4, y4]]))
    M_inv = np.linalg.inv(M)

    print(x_vec)
    a_vec =  M_inv @ x_vec

    return a_vec


def projektive_equalization(image, a_vec):
    equal_image = np.zeros(image.shape)

    a1 = a_vec[0][0]
    a2 = a_vec[1][0]
    a3 = a_vec[2][0]
    b1 = a_vec[3][0]
    b2 = a_vec[4][0]
    b3 = a_vec[5][0]
    c1 = a_vec[6][0]
    c2 = a_vec[7][0]

    for x in range(equal_image.shape[0]):
        for y in range(equal_image.shape[1]):

            denuminator = ( (b1*c2 - b2*c1)*x + (a2*c1 - a1*c2)*y + a1*b2 - a2*b1 )
            
            x_ = int( np.rint(( (b2 - c2*b3)*x + (a3*c2 - a2)*y + a2*b3 - a3*b2 ) / denuminator ))
            y_ = int( np.rint(( (b3*c1 - b1)*x + (a1 - a3*c1)*y + a3*b1 - a1*b3 ) / denuminator ))

            if ( 0 < x_ <= equal_image.shape[0]-1 and 0 < y_ < equal_image.shape[1]-1):
                equal_image[x, y, :] = image[x_, y_, :]

    return equal_image

if __name__ == "__main__":
    # Load Image
    image = skimage.io.imread(fname="./Übung2/schraegbild_tempelhof.jpg")

    """
    picture_points = [[367,334], [344, 434], [521,331], [653, 427]]
    # object_points = [[52.471599, 13.416611],[52.471024, 13.391926], [52.474219, 13.389994], [52.475361, 13.416062]]
    object_points = [[160, 154], [187, 840], [304, 101], [357, 822]]
    """

    #passpoints =                       np.array([[[338, 345], [100, 250]]])
    #passpoints = np.append(passpoints, np.array([[[432, 313], [657, 250]]]), axis = 0)
    #passpoints = np.append(passpoints, np.array([[[335, 545], [100, 610]]]), axis = 0)
    #passpoints = np.append(passpoints, np.array([[[423, 681], [657, 610]]]), axis = 0)


    picture_points = [[338, 345],[432, 313],[335, 545],[423, 681]]
    object_points = [[100, 250], [657, 250], [100, 610], [657, 610]]


    a_vec = projective_qualization_parameter(object_points, picture_points)

    print(a_vec)

    equal_image = projektive_equalization(image, a_vec)

    fig = plt.figure(figsize=(1, 2))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)

    fig.add_subplot(1, 2, 2)
    plt.imshow(equal_image.astype(np.uint8))

    plt.show()