import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage.color import rgb2gray

def projective_qualization_parameter(object_points, picture_points):
    M = np.zeros(shape=(2*len(object_points), 8)).astype('int')
    x_vec = np.zeros(shape=(2*len(object_points), 1)).astype('int')

    for point_nr in range(len(object_points)):
        picture_point = picture_points[point_nr]
        object_point = object_points[point_nr]

        x = object_point[0]
        y = object_point[1]
        x_ = picture_point[0]
        y_ = picture_point[1]

        index = point_nr*2
   
        M[index] = np.array([x_, y_, 1, 0, 0, 0, -x*x_, -x*y_])
        M[index + 1] = np.array([0, 0, 0, x_, y_, 1, -y*x_, -y*y_])

        x_vec[index] = [x]
        x_vec[index + 1] = [y]

    M_inv = np.linalg.inv(M)
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
    
    picture_points = [[345, 434],[653, 427],[521,332],[366, 335]]
    object_points = [[187, 839], [358, 822], [305, 102], [160, 155]]
        """

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