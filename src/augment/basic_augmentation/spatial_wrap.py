# import cv2
# import numpy as np
# import math
#
# img = cv2.imread('frame_1.jpg', cv2.IMREAD_GRAYSCALE)
# rows, cols = img.shape
#
# #####################
# # Vertical wave
#
# img_output = np.zeros(img.shape, dtype=img.dtype)
#
# for i in range(rows):
#     for j in range(cols):
#         offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
#         offset_y = 0
#         if j+offset_x < rows:
#             img_output[i,j] = img[i,(j+offset_x)%cols]
#         else:
#             img_output[i,j] = 0
#
# cv2.imshow('Input', img)
# cv2.imshow('Vertical wave', img_output)
#
# #####################
# # Horizontal wave
#
# img_output = np.zeros(img.shape, dtype=img.dtype)
#
# for i in range(rows):
#     for j in range(cols):
#         offset_x = 0
#         offset_y = int(16.0 * math.sin(2 * 3.14 * j / 150))
#         if i+offset_y < rows:
#             img_output[i,j] = img[(i+offset_y)%rows,j]
#         else:
#             img_output[i,j] = 0
#
# cv2.imshow('Horizontal wave', img_output)
#
# #####################
# # Both horizontal and vertical
#
# img_output = np.zeros(img.shape, dtype=img.dtype)
#
# for i in range(rows):
#     for j in range(cols):
#         offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
#         offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
#         if i+offset_y < rows and j+offset_x < cols:
#             img_output[i,j] = img[(i+offset_y)%rows,(j+offset_x)%cols]
#         else:
#             img_output[i,j] = 0
#
# cv2.imshow('Multidirectional wave', img_output)
#
# #####################
# # Concave effect
#
# img_output = np.zeros(img.shape, dtype=img.dtype)
#
# for i in range(rows):
#     for j in range(cols):
#         offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2*cols)))
#         offset_y = 0
#         if j+offset_x < cols:
#             img_output[i, j] = img[i,(j+offset_x)%cols]
#         else:
#             img_output[i, j] = 0
#
# cv2.imshow('Concave', img_output)
#
# cv2.waitKey()

from PIL import Image
import math
import cv2
import numpy as np
import torch

def vector_length(vector):
  return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


def points_distance(point1, point2):
  return vector_length((point1[0] - point2[0],point1[1] - point2[1]))


def clamp(value, minimum, maximum):
  return max(min(value,maximum),minimum)

## Warps an image accoording to given points and shift vectors.
#
#  @param image input image
#  @param points list of (x, y, dx, dy) tuples
#  @return warped image


def spatial_warp(image_pixels, point):
  result = np.zeros((image_pixels.shape), dtype=np.uint8)
  for y in range(image_pixels.shape[1]):
    for x in range(image_pixels.shape[0]):
        offset = [0,0]
        point_position = (point[0] + point[2], point[1] + point[3])
        shift_vector = (point[2], point[3])
        helper = 1.0 / (3 * (points_distance((x, y), point_position) / vector_length(shift_vector)) ** 4 + 1)

        offset[0] -= helper * shift_vector[0]
        offset[1] -= helper * shift_vector[1]
        coords = (clamp(x + int(offset[0]), 0, image_pixels.shape[0] - 1), clamp(y + int(offset[1]),0, image_pixels.shape[1] - 1))
        result[x,y] = image_pixels[coords[0], coords[1]]
  return result


def spatial_warp_torch(images_tensor, point):
    temp = images_tensor.permute(0, 2, 3, 4, 1)
    # print(temp.size()) # 16 x 224 x 224 x 3
    result = temp.clone()
    for y in range(max(0, point[1] - 3 * point[3]), min(temp.size(2), point[1] + 3 * point[3])):
        for x in range(max(0, point[0] - 3 * point[2]), min(temp.size(3), point[0] + 3 * point[2])):
            offset = [0, 0]
            point_position = (point[0] + point[2], point[1] + point[3])
            shift_vector = (point[2], point[3])
            helper = 0.5 / (3 * (points_distance((x, y), point_position) / vector_length(shift_vector)) ** 4 + 1)
            # print(helper)
            offset[0] -= helper * shift_vector[0]
            offset[1] -= helper * shift_vector[1]
            coords = (clamp(x + int(offset[0]), 0, temp.size(2) - 1), clamp(y + int(offset[1]), 0, temp.size(3) - 1))
            result[:, :, x, y] = temp[:, :, coords[0], coords[1]]
    return result.permute(0, 4, 1, 2, 3)


if __name__ == '__main__':
    image = cv2.imread("frame_1.jpg")
    dim = (256, 256)
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('origin', resized)
    # warp_image = warp(image, [(210,296,100,0), (101,97,-30,-10), (77,473,50,-100)])
    warp_image = spatial_warp(resized, (210, 296, 100, 100))
    cv2.imshow('warped', warp_image)
    cv2.imshow('diff', warp_image - resized)
    cv2.waitKey(0)