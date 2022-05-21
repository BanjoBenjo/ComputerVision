import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage.color import rgb2gray
from PIL import ImageFilter
from scipy.ndimage import gaussian_filter


def projective_equalization_parameter(object_points, picture_points):
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
            if denuminator == 0:
                break
            
            x_ = int( np.rint(( (b2 - c2*b3)*x + (a3*c2 - a2)*y + a2*b3 - a3*b2 ) / denuminator ))
            y_ = int( np.rint(( (b3*c1 - b1)*x + (a1 - a3*c1)*y + a3*b1 - a1*b3 ) / denuminator ))

            if ( 0 < x_ <= equal_image.shape[0]-1 and 0 < y_ < equal_image.shape[1]-1):
                equal_image[x, y, :] = image[x_, y_, :]

    return equal_image

def calculate_weights(image):
    weights = np.zeros_like(image)

    height = image.shape[0]
    width = image.shape[1]

    M = width
    N = height

    for i in range(N):
        for j in range(M):
            weights[i,j,0] = (1-2/M * np.abs(j-M/2)) * (1-2/N * np.abs(i-N/2))

    return weights


def stitch(img_1, img_2, weights_1, weights_2, opt):
    height = img_1.shape[0]
    width = img_1.shape[1]

    stitched_image = np.zeros_like(img_1)
    stitched_weights = np.zeros_like(weights_1)

    for i in range(height):
        for j in range(width):
            if opt == 'weights':
                if weights_1[i,j] > weights_2[i,j]:
                    stitched_image[i,j,:] = img_1[i,j,:]
                    stitched_weights[i,j] = weights_1[i,j]
                else:
                    stitched_image[i,j,:] = img_2[i,j,:]
                    stitched_weights[i,j] = weights_2[i,j]

            elif opt == 'sum':
                norm = weights_1[i,j]+weights_2[i,j]
                if norm !=0:
                    weights_norm_1=weights_1[i,j]/norm
                    weights_norm_2=weights_2[i,j]/norm
                    stitched_image[i,j,:]= weights_norm_1*img_1[i,j,:]+weights_norm_2*img_2[i,j,:]
                    stitched_weights[i,j] = norm/2

    return (stitched_image, stitched_weights)


def multi_band_blending(img_1, img_1_weights, img_2, img_2_weights,):

    img_1_tp = gaussian_filter(img_1, sigma=1)
    img_2_tp = gaussian_filter(img_2, sigma=1)
    img_1_hp = img_1 - img_1_tp
    img_2_hp = img_2 - img_2_tp

    (stitched_image_tp, w_tp) = stitch(img_1_tp, img_2_tp ,img_1_weights, img_2_weights, 'sum')
    (stitched_image_hp, w_hp) = stitch(img_1_hp, img_2_hp ,img_1_weights, img_2_weights, 'weights')


    stitched_image = stitched_image_tp + stitched_image_hp # addieren der Bilder

    return stitched_image, w_tp

if __name__ == "__main__":
    # Load Image
    number_images = 4
    images = [None] * number_images
    picture_points = [None] * number_images
    object_points = [None] * number_images

    equalized_images = [None] * number_images
    weights = [None] * number_images

    images[0] = skimage.io.imread(fname="./Übung3/IMG_1.JPG")
    images[1] = skimage.io.imread(fname="./Übung3/IMG_2.JPG")
    images[2] = skimage.io.imread(fname="./Übung3/IMG_3.JPG")
    images[3] = skimage.io.imread(fname="./Übung3/IMG_4.JPG")

    shape = np.shape(images[0])

    picture_points[0] = [[148, 247],[130, 516],[665, 534],[642, 294]]
    object_points[0] = [[105, 5], [105, 100], [295, 100], [295, 5]]

    picture_points[1] = [[114, 215],[133, 565],[674, 547],[708, 237]]
    object_points[1] = [[100, 120], [100, 230], [295, 230], [295, 120]]

    picture_points[2] = [[140, 373],[164, 635],[632, 619],[690, 379]]
    object_points[2] = [[100, 240], [100, 350], [295, 350], [295, 240]]

    picture_points[3] = [[182, 344],[206, 502],[569, 493],[633, 343]]
    object_points[3] = [[100, 360], [100, 470], [295, 470], [295, 360]]

    init_weights = calculate_weights(np.zeros(shape))


    #for image_nr in range(len(images)):
    for image_nr in range(4):
        z=2
        object_points_scaled = np.array([[y*z, x*z] for y,x in object_points[image_nr]])

        a_vec = projective_equalization_parameter(object_points_scaled, picture_points[image_nr])
        equal_image = projektive_equalization(images[image_nr], a_vec)

        equal_weights = projektive_equalization(init_weights, a_vec)

        equalized_images[image_nr] = equal_image
        weights[image_nr] = np.squeeze(equal_weights[:,:,0]) 


    (stitched_image_01, stitched_weights_01) = stitch(equalized_images[0], equalized_images[1], weights[0], weights[1], 'weights')
    (stitched_image_012, stitched_weights_012) = stitch(stitched_image_01, equalized_images[2], stitched_weights_01, weights[2], 'weights')
    (stitched_image_0123, stitched_weights_0123) = stitch(stitched_image_012, equalized_images[3], stitched_weights_012, weights[3], 'weights')

    (stitched_image_sum_01, stitched_weights_sum_01) = stitch(equalized_images[0], equalized_images[1], weights[0], weights[1], 'sum')
    (stitched_image_sum_012, stitched_weights_sum_012) = stitch(stitched_image_sum_01, equalized_images[2], stitched_weights_sum_01, weights[2], 'sum')
    (stitched_image_sum_0123, stitched_weights_sum_0123) = stitch(stitched_image_sum_012, equalized_images[3], stitched_weights_sum_012, weights[3], 'sum')

    (mbb_image1, mbb_weights1) = multi_band_blending(equalized_images[0], weights[0], equalized_images[1], weights[1])
    (mbb_image2, mbb_weights2) = multi_band_blending(mbb_image1, mbb_weights1, equalized_images[2], weights[2])
    (mbb_image3, mbb_weights3) = multi_band_blending(mbb_image2, mbb_weights2, equalized_images[3], weights[3])

    for image_nr in range(len(images)):
        fig = plt.figure(figsize=(1, 2))
        fig.add_subplot(1, 2, 1)
        plt.imshow(equalized_images[image_nr].astype(np.uint8))

        fig.add_subplot(1, 2, 2)
        plt.imshow(weights[image_nr], interpolation='none')

    fig = plt.figure()
    plt.imshow(stitched_image_0123.astype(np.uint8))

    fig = plt.figure()
    plt.imshow(stitched_image_sum_0123.astype(np.uint8))

    fig = plt.figure()
    plt.imshow(mbb_image3.astype(np.uint8))
    
    plt.show()