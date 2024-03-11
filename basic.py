from curses import erasechar
import time
import math
import pdb
import numpy as np
from jax import jit, grad, jacfwd, jacrev
from matplotlib import pyplot as plt
import scipy.stats as stats
import statistics
from cyipopt import minimize_ipopt
import jax.numpy as jnp


def rotationMatrix(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def rotationMatrixJ(theta):
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0],
            [jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# ray is [point, theta]
# point is [x, y]
# fov is 35 in both directions
fov = 70 * np.pi / 180
# 11 width
image1 = [[30.5, 0], [41.5, 0]]
image2 = [[0, 78.5], [0, 89.5]]
image1Sim = [[55.0, -20], [65.0, -20]]
image2Sim = [[55.0, 100], [65.0, 100]]
# images = [image1, image2]
images = []

actual_measurements = []
prediced_measurements = []
measured_var = []
predicted_var = []

# a = 1000
# testMat = np.array([[1, 1, -a],
#                    [1, -1, a],
#                    [1, -1, -a],
#                    [1, 1, a]])
# testInv = np.array([[0.25, 0, 0.5, 0.25],
#                     [-0.25, 0.5, 0, -0.25],
#                     [0, 0.5/a, -0.5/a, 0]])
# print("psuedo inv" + str(np.linalg.pinv(testMat)))
# print("test inv:" + str(testInv @ testMat))


def inBetween(a, n1, n2):
    return (a >= n1 and a <= n2) or (a <= n1 and a >= n2)


def doublesEqual(a, b):
    return np.abs(a - b) < 0.0000001


def angleWrap(angle):
    return angle % (2 * np.pi)


def quadraticFormula(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return []
    elif doublesEqual(discriminant, 0):
        return [-b / (2 * a)]
    else:
        return [
            (-b + np.sqrt(discriminant)) / (2 * a),
            (-b - np.sqrt(discriminant)) / (2 * a),
        ]


def detectsPoint(robot_state, point):
    robot_angle = robot_state[2][0]
    offset = np.pi - robot_angle
    angleToPoint = math.atan2(
        point[1] - robot_state[1][0], point[0] - robot_state[0][0]
    )
    offset_point_angle = angleWrap(angleToPoint + offset)
    min_angle = angleWrap(robot_angle - fov / 2 + offset)
    max_angle = angleWrap(robot_angle + fov / 2 + offset)
    return (offset_point_angle >= min_angle) and (offset_point_angle <= max_angle)


def detectsImage(robot_state, image):
    # image is 2 points [[x1, y1], [x2, y2]]
    return detectsPoint(robot_state, image[0]) and detectsPoint(robot_state, image[1])


def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def centerOfImage(image):
    # [[x1, y1], [x2, y2]]
    return [(image[0][0] + image[1][0]) / 2, (image[0][1] + image[1][1]) / 2]


def findBestImage(robot_state):
    minDistance = 99999999999
    closestImage = [[0, 0], [0, 0]]
    foundImage = False
    for image in images:
        # print(image)
        if detectsImage(robot_state, image):
            # print("detectsImage done")
            distanceToImage = distance(
                np.array(robot_state).T[0],
                [(image[0][0] + image[1][0]) / 2, (image[0][1] + image[1][1]) / 2],
            )
            if distanceToImage < minDistance:
                minDistance = distanceToImage
                closestImage = image
                foundImage = True
    return [closestImage, foundImage]
