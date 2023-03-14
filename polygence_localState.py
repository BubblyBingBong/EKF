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
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
def rotationMatrixJ(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0],
                   [jnp.sin(theta), jnp.cos(theta), 0],
                   [0, 0, 1]])

#ray is [point, theta]
#point is [x, y]
#fov is 35 in both directions
fov = 70 * np.pi/180
#11 width
image1 = [[30.5, 0], [41.5, 0]]
image2 = [[0, 78.5], [0, 89.5]]
image1Sim = [[55.0, -20], [65.0, -20]]
image2Sim = [[55.0, 100], [65.0, 100]]
#images = [image1, image2]
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
    return np.abs(a - b) < .0000001

def angleWrap(angle):
    return angle % (2*np.pi)

def quadraticFormula(a, b, c):
    discriminant = b ** 2 - 4*a*c
    if (discriminant < 0):
        return []
    elif (doublesEqual(discriminant, 0)):
        return [-b / (2 * a)]
    else:
        return [(-b + np.sqrt(discriminant))/(2 * a), (-b - np.sqrt(discriminant))/(2 * a)]
    

def detectsPoint(robot_state, point):
    robot_angle = robot_state[2][0]
    offset = np.pi - robot_angle
    angleToPoint = math.atan2(point[1] - robot_state[1][0], point[0] - robot_state[0][0])
    offset_point_angle = angleWrap(angleToPoint + offset)
    min_angle = angleWrap(robot_angle - fov/2 + offset)
    max_angle = angleWrap(robot_angle + fov/2 + offset)
    return (offset_point_angle >= min_angle) and (offset_point_angle <= max_angle)

def detectsImage(robot_state, image):
    #image is 2 points [[x1, y1], [x2, y2]]
    return detectsPoint(robot_state, image[0]) and detectsPoint(robot_state, image[1])

def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def centerOfImage(image):
    #[[x1, y1], [x2, y2]]
    return [(image[0][0] + image[1][0])/2, (image[0][1] + image[1][1])/2]

def findBestImage(robot_state):
    minDistance = 99999999999
    closestImage = [[0, 0], [0, 0]]
    foundImage = False;
    for image in images:
        #print(image)
        if (detectsImage(robot_state, image)):
            #print("detectsImage done")
            distanceToImage = distance(np.array(robot_state).T[0], [(image[0][0] + image[1][0])/2, (image[0][1] + image[1][1])/2])
            if (distanceToImage < minDistance):
                minDistance = distanceToImage
                closestImage = image
                foundImage = True;
    return [closestImage, foundImage]
    
L_dis = 5
l_dis = 6.5
r_dis = 1
W_dis = 2
D_dis = 1
class RobotActual:
    def __init__(self, dt, measureDt):
        self.dt = dt
        self.measureDt = measureDt
        self.states = np.zeros([1, 6])#np.array([[0., 0., 0., 0., 0., 0.]])
        #print("size: " + str(len(self.states)))
        self.measured_states = np.zeros([1, 6])
        self.firstUpdate = True
        self.firstMeasureUpdate = True
    def reset(self):
        self.states = np.zeros([1, 6])#np.array([[0., 0., 0., 0., 0., 0.]])
        #print("size: " + str(len(self.states)))
        self.measured_states = np.zeros([1, 6])
        self.firstUpdate = True
        self.firstMeasureUpdate = True
    def step(self, u):
        prev_state = self.getPrevState()
        local_vel = np.array([[prev_state[3], prev_state[4], prev_state[5]]]).T
        global_vel = rotationMatrix(prev_state[2]) @ local_vel
        
        xPos = prev_state[0] + self.dt * global_vel[0][0]
        yPos = prev_state[1] + self.dt * global_vel[1][0]
        theta = prev_state[2] + self.dt * global_vel[2][0]
        M = (1/r_dis) * np.array([[1, -1, -(L_dis + l_dis)],
                              [1,  1,   L_dis + l_dis],
                              [1,  1, -(L_dis + l_dis)],
                              [1, -1,   L_dis + l_dis]])
        M_inv = np.linalg.pinv(M)
        M_inv_u = M_inv @ u
        A = M_inv_u[0][0]
        B = M_inv_u[1][0]
        C = M_inv_u[2][0]
        xVel = A
        yVel = B
        thetaVel = C        
        Q = np.array([[self.dt * 2E-3,  0,  0,  0,  0,  0],
                      [0, self.dt * 2E-3,  0,  0,  0,  0],
                      [0,  0, self.dt * 2E-3,  0,  0,  0],
                      [0,  0,  0, 4.5E-1,  0,  0],
                      [0,  0,  0,  0, 4.5E-1,  0],
                      [0,  0,  0,  0,  0, 4.5E-1]])
        #Q = np.zeros((6, 6)) #MPC
        q = np.array([[np.sqrt(Q[0][0]) * np.random.randn()],
                      [np.sqrt(Q[1][1]) * np.random.randn()],
                      [np.sqrt(Q[2][2]) * np.random.randn()],
                      [np.sqrt(Q[3][3]) * np.random.randn()],
                      [np.sqrt(Q[4][4]) * np.random.randn()],
                      [np.sqrt(Q[5][5]) * np.random.randn()]])

        reality = [[xPos, yPos, theta, xVel, yVel, thetaVel]] + q.T
        if (self.firstUpdate):
            self.states = np.array(reality)
        else:
            self.states = np.append(self.states, reality, axis=0)
        self.firstUpdate = False

    def stateToMeasurement(self, state):
        theta = state[2]
        xVel = state[3]
        yVel = state[4]
        thetaVel = state[5]
        
        a = yVel - D_dis * thetaVel
        R = xVel + W_dis * thetaVel
        L = xVel - W_dis * thetaVel
        return np.array([[state[0], state[1], state[2], L, R, a]]).T

    def measurementToLocalVel(self, measurement):
        L = measurement[3][0]
        R = measurement[4][0]
        a = measurement[5][0]

        local_xVel = ((L + R)/2)
        local_thetaVel = ((R - L)/(2 * W_dis))
        local_yVel = (a + D_dis * local_thetaVel)

        local_vel = np.array([[local_xVel, local_yVel, local_thetaVel]]).T

        return local_vel
        
    def measure(self):
        #odometry measurement
        actual_state = self.getPrevState()
        measurement = self.stateToMeasurement(actual_state)
        #measured_localVel = self.measurementToLocalVel(measurement)
        #measured_globalVel = rotationMatrix(actual_state[2]) @ measured_localVel
        #actual_measurements.append(measured_globalVel.T[0])
        theta = actual_state[2] 
        xVel = actual_state[3]
        yVel = actual_state[4]
        thetaVel = actual_state[5]
        
        [bestImage, detected] = findBestImage(np.array([actual_state]).T)
        d = distance(actual_state, centerOfImage(bestImage))
        #print(measurement)

        R = np.array([[1E-3 * d ** 2 + 1E-3, 0, 0, 0, 0, 0],
                     [0, 1E-3 * d ** 2 + 1E-3, 0, 0, 0, 0],
                     [0, 0, 1E-3 * d ** 2 + 1E-3, 0, 0, 0],
                     [0, 0, 0, 2E-3 * measurement[3][0] ** 2 + 1E-3, 0, 0],
                     [0, 0, 0, 0, 2E-3 * measurement[4][0] ** 2 + 1E-3, 0],
                     [0, 0, 0, 0, 0, 2E-3 * measurement[5][0] ** 2 + 1E-3]])
        # R[3][3] = 0.
        # R[4][4] = 0.
        # R[5][5] = 0.
        #print(R[3])
        r = np.array([[np.sqrt(R[0][0]) * np.random.randn()],
                      [np.sqrt(R[1][1]) * np.random.randn()],
                      [np.sqrt(R[2][2]) * np.random.randn()],
                      [np.sqrt(R[3][3]) * np.random.randn()],
                      [np.sqrt(R[4][4]) * np.random.randn()],
                      [np.sqrt(R[5][5]) * np.random.randn()]])
        
        noisy_measurement = measurement + r
        #print(measurement.T)
        return noisy_measurement
        # prev_measurement = self.getPrevMeasurement()
        # localVel = self.measurementToLocalVel(noisy_measurement)
        # globalVel = rotationMatrix(theta) @ localVel
        # new_measurement = np.array([[prev_measurement[0] + self.measureDt*prev_measurement[3],
        #                              prev_measurement[1] + self.measureDt*prev_measurement[4],
        #                              prev_measurement[2] + self.measureDt*prev_measurement[5],
        #                              globalVel[0][0], globalVel[1][0], globalVel[2][0]]])
        # self.measured_states = np.append(self.measured_states, new_measurement, axis = 0)
        #print(self.measured_states)
        
    
    def getPrevState(self):
        if (self.firstUpdate):
            return initial_state
        else:
            return self.states[-1]

    def getPrevMeasurement(self):
        if (self.firstMeasureUpdate):
            self.firstMeasureUpdate = False
            return initial_state
        else:
            return self.measured_states[-1]

class RobotEKF:
    def __init__(self, dt, mode):
        self.mode = mode
        self.P = np.zeros([6, 6])
        self.dt = dt
        self.states = np.zeros([1, 6])#np.array([[0., 0., 0., 0., 0., 0.]])
        self.predicted_states = np.zeros([1, 6]) #only prediction
        self.A = np.zeros([6, 6])
        self.firstUpdate = True
        self.firstPredictUpdate = True
        #print("initial state: " + str(self.states))

        #based off prev state
        self.predictions = np.array([initial_state])
        self.measurements = np.array([initial_state])

        self.raw_measurements = np.array([[0, 0, 0, 0, 0, 0]])
        self.predicted_measurements = np.array([[0, 0, 0, 0, 0, 0]])
    def reset(self):
        self.P = np.zeros([6, 6])
        self.states = np.zeros([1, 6])
        self.predicted_states = np.zeros([1, 6])
        self.A = np.zeros([6, 6])
        self.firstUpdate = True
        self.firstPredictUpdate = True

        self.predictions = np.array([initial_state])
        self.measurements = np.array([initial_state])

        self.raw_measurements = np.array([[0, 0, 0, 0, 0, 0]])
        self.predicted_measurements = np.array([[0, 0, 0, 0, 0, 0]])
    def stateToMeasurement(self, state):
        theta = state[2]
        xVel = state[3]
        yVel = state[4]
        thetaVel = state[5]
        
        a = yVel - D_dis * thetaVel
        R = xVel + W_dis * thetaVel
        L = xVel - W_dis * thetaVel
        return np.array([[state[0], state[1], state[2], L, R, a]]).T
    
    def step(self, u, measurement):
        prev_state = self.getPrevState()
        #prev_predicted_state = self.getPrevPredicted()
        
        [prediction, Q] = self.predict(u, prev_state)
        #self.predictions = np.append(self.predictions, prediction.T, axis = 0)
        #[only_prediction, Q2] = self.predict(u, prev_predicted_state)
        #self.predicted_states = np.append(self.predicted_states, only_prediction.T, axis = 0)
        #measured_states = np.append(measured_states, measurement.T, axis = 0)
        
        #self.raw_measurements = np.append(self.raw_measurements, measurement.T, axis = 0)
        
        [bestImage, detected] = findBestImage(prediction)
        if (detected):
            d = distance(prediction.T[0], centerOfImage(bestImage))
        else:
            d = 1E+25 #large noise to simulate image not being detected
        #print("done detect")
        #print(0.001 * d ** 2 + 0.00001)
        R = np.array([[1E-3 * d ** 2 + 1E-3, 0, 0, 0, 0, 0],
                     [0, 1E-3 * d ** 2 + 1E-3, 0, 0, 0, 0],
                     [0, 0, 1E-3 * d ** 2 + 1E-3, 0, 0, 0],
                     [0, 0, 0, 2E-3 * measurement[3][0] ** 2 + 1E-3, 0, 0],
                     [0, 0, 0, 0, 2E-3 * measurement[4][0] ** 2 + 1E-3, 0],
                     [0, 0, 0, 0, 0, 2E-3 * measurement[5][0] ** 2 + 1E-3]])
        # R[3][3] = 0.
        # R[4][4] = 0.
        # R[5][5] = 0.
        if self.mode == 1:
            for i in range(3, 6):
                R[i][i] = 99999999.

        #measured_var.append(np.trace(R))
        #predicted_var.append(np.trace(Q))
        measured_var.append(R[3][3] + R[4][4] + R[5][5])
        predicted_var.append(Q[3][3] + Q[4][4] + Q[5][5])
        
        #print("R" + str(R[3][3]))
        # theta = prediction[2][0]
        # #print(theta)
        # xVel = prediction[3][0]
        # yVel = prediction[4][0]
        # thetaVel = prediction[5][0]
        # mdt = self.dt
        H = np.array([[1.0, 0, 0, 0, 0, 0],
                      [0, 1.0, 0, 0, 0, 0],
                      [0, 0, 1.0, 0, 0, 0],
                      [0, 0, 0, 1.0, 0, -W_dis],
                      [0, 0, 0, 1.0, 0, W_dis],
                      [0, 0, 0, 0, 1.0, -D_dis]])
        A = self.A
        P = self.P
        predicted_measurement = self.stateToMeasurement(prediction.T[0])
        #self.predicted_measurements = np.append(self.predicted_measurements, predicted_measurement.T, axis=0)
        
        P_prediction = A @ P @ A.T + Q
        
        K = np.linalg.solve((H @ P_prediction @ H.T + R).T, (P_prediction @ H.T).T).T
        #K = P_prediction @ H.T @ np.linalg.inv(H @ P_prediction @ H.T + R)
        estimate = prediction + K @ (measurement - predicted_measurement)
        self.P = P_prediction - K @ H @ P_prediction

        if (self.firstUpdate):
            self.states = np.array(estimate.T)
        else: 
            self.states = np.append(self.states, np.array(estimate.T), axis=0)
        self.firstUpdate = False

        return estimate

    def predict(self, u, prev_state):
        Q = np.array([[self.dt * 2E-3,  0,  0,  0,  0,  0],
                      [0, self.dt * 2E-3,  0,  0,  0,  0],
                      [0,  0, self.dt * 2E-3, 0,  0,  0],
                      [0,  0,  0, 4.5E-1, 0,  0],
                      [0,  0,  0,  0, 4.5E-1, 0],
                      [0,  0,  0,  0,  0, 4.5E-1]])
        if (self.mode == 2):
            Q[3][3] = 9999999999.
            Q[4][4] = 9999999999.
            Q[5][5] = 9999999999.

        #Q = np.zeros((6, 6)) #MPC
        #Q = np.zeros((6, 6)) #MPC
        #print(prev_state)
        local_vel = np.array([[prev_state[3], prev_state[4], prev_state[5]]]).T
        global_vel = rotationMatrix(prev_state[2]) @ local_vel
        
        xPos = prev_state[0] + self.dt * global_vel[0][0]
        yPos = prev_state[1] + self.dt * global_vel[1][0]
        theta = prev_state[2] + self.dt * global_vel[2][0]
        M = (1/r_dis) * np.array([[1, -1, -(L_dis + l_dis)],
                              [1,  1,   L_dis + l_dis],
                              [1,  1, -(L_dis + l_dis)],
                              [1, -1,   L_dis + l_dis]])
        M_inv = np.linalg.pinv(M)
        M_inv_u = M_inv @ u
        c1 = M_inv_u[0][0]
        c2 = M_inv_u[1][0]
        c3 = M_inv_u[2][0]
        local_xVel = c1
        local_yVel = c2
        thetaVel = c3
        dt = self.dt
        self.A = np.array([[1, 0, dt*(-prev_state[3]*np.sin(theta) - prev_state[4]*np.cos(theta)), dt * np.cos(theta), -dt * np.sin(theta), 0],
                           [0, 1, dt*(prev_state[3]*np.cos(theta) - prev_state[4]*np.sin(theta)), dt * np.sin(theta), dt * np.cos(theta), 0],
                           [0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])
       
        prediction = np.array([[xPos, yPos, theta, local_xVel, local_yVel, thetaVel]]).T
        #print(prediction[5][0])
        return [prediction, Q]
   
    def getPrevState(self):
        if (self.firstUpdate):
            #print("forst")
            return initial_state
        else:
            return self.states[-1]
        
    def getPrevPredicted(self):
        if (self.firstPredictUpdate):
            self.firstPredictUpdate = False
            return initial_state
        else:
            return self.predicted_states[-1]

class PID():
    def __init__(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D
        self.prevError = 0
        self.errorSum = 0
    def step(self, error):
        self.errorSum += error
        p = self.P * error
        i = self.I * self.errorSum
        d = self.D * (error - self.prevError)
        self.prevError = error
        return p + i + d
    def reset(self):
        self.errorSum = 0
        self.prevError = 0

class Circle():
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    def getLineIntersections(self, line):
        #the line is still a segment class
        h = self.center[0]
        k = self.center[1]
        r = self.radius
        x1 = line.startPoint[0]
        y1 = line.startPoint[1]
        x2 = line.endPoint[0]
        y2 = line.endPoint[1]

        intersections = []
        if (line.isVertical()):
            y_values = quadraticFormula(1, - 2 * k, (x1 - h) ** 2 + k*k - r*r)
            #print("y values: " + str(y_values))
            for y_val in y_values:
                intersections.append([x1, y_val])
        else:
            m = (y2 - y1)/(x2 - x1)
            a = 1 + m*m
            b = -2*h - 2*m*m*x1 + 2*m*y1 - 2*m*k
            c = h*h + m*m*x1*x1 - 2*m*x1*y1 + 2*m*x1*k + (y1-k) ** 2 - r*r
            x_values = quadraticFormula(a, b, c)
            for x_val in x_values:
                intersections.append([x_val, line.function(x_val)])
        return intersections
    def getSegmentIntersections(self, segment):
        lineIntersections = self.getLineIntersections(segment)
        intersections = []
        for intersection in lineIntersections:
            if (segment.inSegmentRange(intersection)):
                intersections.append(intersection)
        return intersections
    def getPathEndIntersections(self, ray):
        lineIntersections = self.getLineIntersections(ray)
        intersections = []
        for intersection in lineIntersections:
            if (ray.inRayRange(intersection)):
                if(ray.inSegmentRange(intersection)):
                    intersections.append(intersection)
                else:
                    intersections.append(ray.endPoint)
        return intersections
            
        
class Segment():
    def __init__(self, startPoint, endPoint):
        self.startPoint = startPoint
        self.endPoint = endPoint
    def isVertical(self):
        return doublesEqual(self.startPoint[0], self.endPoint[0])
    def getAngle(self):
        radians = 0
        if (self.isVertical()):
            if (self.endPoint[1] > self.startPoint[1]):
                return np.pi/2
            else:
                return -np.pi/2
        else:
            return math.atan2(self.endPoint[1] - self.startPoint[1], self.endPoint[0] - self.startPoint[0])
    def moveAlongSegment(self, distance):
        angle = self.getAngle()
        return [self.startPoint[0] + distance * np.cos(angle), self.startPoint[1] + distance * np.sin(angle)]
    def function(self, x):
        m = (self.endPoint[1] - self.startPoint[1])/(self.endPoint[0] - self.startPoint[0])
        return m * (x - self.startPoint[0]) + self.startPoint[1]
    def inSegmentRange(self, point):
        return inBetween(point[0], self.startPoint[0], self.endPoint[0]) and inBetween(point[1], self.startPoint[1], self.endPoint[1])
    def inRayRange(self, point):
        if (self.endPoint[0] > self.startPoint[0]):
            withinX = point[0] >= self.startPoint[0]
        else:
            withinX = point[0] <= self.startPoint[0]

        if (self.endPoint[1] > self.startPoint[1]):
            withinY = point[1] >= self.startPoint[1]
        else:
            withinY = point[1] <= self.startPoint[1]
        return withinX and withinY
    def closestToEnd(self, points):
        if (len(points) == 0):
            return [0, 0]
        minDistance = 999999999
        closestPoint = points[0]
        for point in points:
            if (distance(point, self.endPoint) < minDistance):
                minDistance = distance(point, self.endPoint)
                closestPoint = point
        return closestPoint
    def lineIntersection(self, line):
        self_x1 = self.startPoint[0]
        self_y1 = self.startPoint[1]
        self_x2 = self.endPoint[0]
        self_y2 = self.endPoint[1]
        
        other_x1 = line.startPoint[0]
        other_y1 = line.startPoint[1]
        other_x2 = line.endPoint[0]
        other_y2 = line.endPoint[1]
        if (self.isVertical() and line.isVertical()):
            xVal = 0
            yVal = 0
        elif (not(self.isVertical()) and line.isVertical()):
            xVal = line.startPoint[0]
            yVal = (self_y2 - self_y1)/(self_x2 - self_x1) * (xVal - self_x1) + self_y1
        elif (self.isVertical() and not(line.isVertical())):
            xVal = self.startPoint[0]
            yVal = (other_y2 - other_y1)/(other_x2 - other_x1) * (xVal - other_x1) + other_y1
        else: #both not vertical
            self_slope = (self_y2 - self_y1)/(self_x2 - self_x1)
            other_slope = (other_y2 - other_y1)/(other_x2 - other_x1)
            if (doublesEqual(self_slope, other_slope)):
                return [0, 0]
            self_yint = -self_slope * self_x1 + self_y1
            other_yint = -other_slope * other_x1 + other_y1

            xVal = (other_yint - self_yint)/(self_slope - other_slope)
            yVal = self_slope * xVal + self_yint
        return [xVal, yVal]
    def projectToLine(self, point):
        if (self.isVertical()):
            return [self.startPoint[0], point[1]]
        slope = (self.endPoint[1] - self.startPoint[1])/(self.endPoint[0] - self.startPoint[0])
        if (doublesEqual(slope, 0)):
            return [point[0], self.startPoint[1]]
        perp_slope = -1/slope
        perp_line = Segment(point, [point[0] + 1, point[1] + perp_slope])
        return self.lineIntersection(perp_line)
    def projectToSegment(self, point):
        lineProjection = self.projectToLine(point)
        segmentProjection = lineProjection

        if (lineProjection[0] < min(self.startPoint[0], self.endPoint[0]) or lineProjection[0] > max(self.startPoint[0], self.endPoint[0]) or
            lineProjection[1] < min(self.startPoint[1], self.endPoint[1]) or lineProjection[1] > max(self.startPoint[1], self.endPoint[1])):
            if (distance(lineProjection, self.startPoint) < distance(lineProjection, self.endPoint)):
                segmentProjection = self.startPoint
            else:
                segmentProjection = self.endPoint
        return segmentProjection

class PurePursuit():
    def __init__(self, path, followRadius, xPID, yPID, thetaPID):
        self.path = path
        self.followRadius = followRadius
        self.pathIndex = 1 #endpoint of current segment
        self.xPID = xPID
        self.yPID = yPID
        self.thetaPID = thetaPID
        self.followingEndPoint = False
    def reset(self, path):
        self.pathIndex = 1
        self.xPID.reset()
        self.yPID.reset()
        self.thetaPID.reset()
        self.followingEndPoint = False
    def getTargetPose(self, state):
        followCircle = Circle(state[:2], self.followRadius)
        targetPoint = [123456, 0]
        for i in range(len(self.path) - 1, 0, -1):
            currentSegment = Segment(self.path[i-1], self.path[i])
            if (i == len(self.path) - 1):
                intersections = followCircle.getPathEndIntersections(currentSegment)
            else:
                intersections = followCircle.getSegmentIntersections(currentSegment)
                
            if (len(intersections) > 0):
                targetPoint = currentSegment.closestToEnd(intersections)
                self.pathIndex = np.max([self.pathIndex, i])
                break
        if (doublesEqual(targetPoint[0], 123456)):
            targetPoint = self.path[self.pathIndex][:2]
            
        endPoint = self.path[-1][:2]
        if (not(self.followingEndPoint) and np.sum((endPoint[:2] - targetPoint[:2]) ** 2) < .01):
            self.followingEndPoint = True

        targetPose = [targetPoint[0], targetPoint[1], self.path[self.pathIndex][2]]
        if (self.followingEndPoint):
            #print("following end")
            targetPose = self.path[-1]
        #print("index: " + str(self.pathIndex))
        if (abs(self.path[self.pathIndex][2] - 99999) < 0.1):
            #print("hello bello")
            targetPose[2] = math.atan2(targetPoint[1] - state[1], targetPoint[0] - state[0])

        targets.append(targetPose)
        return targetPose
    
    def step(self, state):
        targetPose = self.getTargetPose(state)
        globalError = np.array([[targetPose[0] - state[0]], [targetPose[1] - state[1]], [targetPose[2] - state[2]]])
        localError = rotationMatrix(-state[2]) @ globalError
        
        xPow = self.xPID.step(localError[0][0])
        yPow = self.yPID.step(localError[1][0])
        thetaPow = self.thetaPID.step(localError[2][0])
        m1 = xPow - yPow - thetaPow
        m2 = xPow + yPow + thetaPow
        m3 = xPow + yPow - thetaPow
        m4 = xPow - yPow + thetaPow
        u = np.array([[m1, m2, m3, m4]]).T
        scaleDown = max(30., max(abs(m1), max(abs(m2), max(abs(m3), abs(m4)))))
        #print(scaleDown)
        u *= (30./scaleDown)
        return u
    def endedPath(self, state):
        return distance(state, self.path[-1]) < 3
    
class MPC():
    def __init__(self, path, horizon, dt):
        self.path = path
        self.horizon = horizon
        self.dt = dt
        self.bnds = jnp.concatenate((jnp.array([(-30, 30)] * 4 * horizon), jnp.array([(-100, 100)] * 6 * horizon)))
        self.prev_optimization = jnp.array([0] * self.horizon * 10)

        self.obj_jit = 0
        self.obj_grad = 0
        self.obj_hess = 0
        self.objJitSetup()

    def predictMPC(self, prev_state, u):
        dt = self.dt
        local_vel = jnp.array([[prev_state[3], prev_state[4], prev_state[5]]]).T
        global_vel = rotationMatrixJ(prev_state[2]) @ local_vel
        
        xPos = prev_state[0] + dt * global_vel[0][0]
        yPos = prev_state[1] + dt * global_vel[1][0]
        theta = prev_state[2] + dt * global_vel[2][0]
        M = (1/r_dis) * jnp.array([[1, -1, -(L_dis + l_dis)],
                          [1, 1, (L_dis + l_dis)], 
                          [1, 1, -(L_dis + l_dis)],
                          [1, -1, (L_dis + l_dis)]])
        M_inv = jnp.linalg.pinv(M)
        M_inv_u = M_inv @ u
        c1 = M_inv_u[0]
        c2 = M_inv_u[1]
        c3 = M_inv_u[2]
        
        xVel = c1 #c1 * jnp.cos(prev_state[2]) - c2 * jnp.sin(prev_state[2])
        yVel = c2# c1 * jnp.sin(prev_state[2]) + c2 * jnp.cos(prev_state[2])
        thetaVel = c3        
        return jnp.array([xPos, yPos, theta, xVel, yVel, thetaVel])

    def objective(self, x, state):
        horizon = self.horizon
        u_list = x[:4 * horizon]
        target_states = self.getTargetTrajectory(state, 60)
        print("target states: " + str(target_states))
        state_cost = 0
        model_constraint = 0
        
        for i in range(horizon):
            print("i: " + str(i))

            if (i >= 1):
                prev_state = x[horizon * 4 + 6 * (i-1) : horizon * 4 + 6 * i]
                current_state = x[horizon * 4 + 6 * i : horizon * 4 + 6 * (i+1)]
                prev_u = x[4 * (i-1) : 4 * i]
                predicted_current = self.predictMPC(prev_state, prev_u)
                
                model_constraint += jnp.sum((predicted_current - current_state) ** 2)
                
            position_state = x[horizon * 4 + i * 6 : horizon * 4 + i * 6 + 3]
            target_position = target_states[i]
            target_position = jnp.array([target_position[0], target_position[1], self.path[-1][2]])
            print("target pos: " + str(target_position))
            print("pos state: " + str(position_state))
            position_cost = jnp.sum((target_position[:2] - position_state[:2]) ** 2)
            angle_cost = 1 * (target_position[2] - position_state[2]) ** 2
            state_cost += (position_cost + angle_cost)

        control_cost = jnp.sum(u_list ** 2)
        
        total_cost = 1 * state_cost + 9999 * model_constraint + 0 * control_cost
        print("cost: " + str(total_cost))
        return total_cost
    
    def getTargetTrajectory(self, prev_state, vel):
        #return jnp.array([[50, 50, 0], [50, 50, 0], [50, 50, 0], [50, 50, 0]])
        
        #really sketchy append
        #'''
        [projection, pathIndex] = self.getClosestPoint(prev_state)
        
        altered_path = self.path
        altered_path = altered_path[pathIndex:]
        arr = [projection]
        for point in altered_path:
            print(point)
            arr.append(point)
        altered_path = arr
        target = self.moveAlongPath(altered_path, vel)
        return target
        #'''
        
    def moveAlongPath(self, path, vel):
        segment_lengths = []
        target_trajectory = []
        atEnd = False
        for i in range(len(path) - 1):
            segment_lengths.append(distance(path[i], path[i+1]))
            
        for i in range(self.horizon):
            segment_distance = vel * (i+1) * self.dt
            j = 0
            while (segment_distance > 0):
                if (j >= len(segment_lengths)):
                    target_trajectory.append(path[-1])
                    #print("at end")
                    atEnd = True
                    break
                segment_distance -= segment_lengths[j]
                j += 1
            if (not(atEnd)):
                segment_distance += segment_lengths[j-1]
                current_segment = Segment(path[j-1], path[j])

                target_trajectory.append(current_segment.moveAlongSegment(segment_distance))
            
        return target_trajectory

    def getClosestPoint(self, prev_state):
        path = self.path
        minDistance = 9999999.
        dis = 0.
        closestPoint = [0, 0]
        pathIndex = 1
        for i in range(len(path) - 1):
            currentSegment = Segment(path[i], path[i+1])
            projection = currentSegment.projectToSegment(prev_state)
            dis = distance(prev_state, projection)
            if (dis < minDistance):
                minDistance = dis
                closestPoint = projection
                pathIndex = i+1
        return [closestPoint, pathIndex]
        
    def objJitSetup(self):
        self.obj_jit = jit(self.objective)
        self.obj_grad = jit(grad(self.obj_jit))
        self.obj_hess = jit(jacrev(jacfwd(self.obj_jit)))

    def step(self, prev_state):
        horizon = self.horizon
        dtMPC = self.dt

        eq_constraints = lambda x, prev_state=prev_state : np.sum((x[4 * horizon : 4 * horizon + 6] - prev_state) ** 2)
        cons_jit = jit(eq_constraints)
        cons_jac = jit(jacfwd(cons_jit))
        cons_hess = jacrev(jacfwd(cons_jit))
        cons_hessvp = jit(lambda x, v, cons_hess=cons_hess: cons_hess(x) * v[0])

        cons = [{'type': 'eq', 'fun': cons_jit, 'jac': cons_jac, 'hess': cons_hessvp}]

        
        # loops = 2
        # test_cat = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        # print("controls test:" + str(test_cat[4:4*loops]))
        # print("states test:" + str(test_cat[4*loops + 6:]))
        # print("optimize test:" + str(np.concatenate((test_cat[4:4*loops], test_cat[4*loops + 6:]), axis=0)))
        
        prev_opt = self.prev_optimization
        cdr_controls = prev_opt[4 : 4*horizon]
        cdr_states = prev_opt[4*horizon+6:]
        x0 = jnp.concatenate((jnp.concatenate((cdr_controls, prev_opt[4*horizon - 4 : 4*horizon])),
                              jnp.concatenate((cdr_states, prev_opt[10*horizon - 6:]))))
        
        res = minimize_ipopt(self.obj_jit, jac=self.obj_grad, hess=self.obj_hess, x0=x0, bounds=self.bnds,
                constraints=cons, options={"max_iter": 300, "acceptable_iter": 1, "constr_viol_tol": 1e-1,
                                           "acceptable_tol": 1e-1, "tol": 1e-2})
        print("")
        print("target trajectory: " + str(self.getTargetTrajectory(prev_state, 120)))
        #print("minimized value: " + str(res.fun))
        optimization = res.x
        print("optimization: " + str(optimization))
        self.prev_optimization = optimization
        #MPC_trajectory = optimization[4 * horizon:]
        #print("mpc trajectory: " + str(MPC_trajectory))
        return np.array([optimization[4:8]]).T
test_arr = [[0, 0], [1, 1], [2, 2]]
test_arr.insert(0, [3, 3])
#print(test_arr)
test_path = [[0, 0], [1, 1], [1, 2]]
testMPC = MPC(test_path, 4, 1)
#print(testMPC.getTargetTrajectory([0.5, 0.5] , 1))

    
targets = []

#print("bnds: " + str(bnds))


#initial_state = np.array([13.75/2, 130, np.pi/2, 0, 0, 0]) #cycle path
mpc_horizon = 4

#paths to test on
CYCLE_HUB = np.array([[13.75/2, 130, np.pi/2], [13.75/2, 80, np.pi/2], [35, 65, 3*np.pi/4]]) #cycle path
CYCLE_WAREHOUSE = np.array([[35, 65, 3*np.pi/4], [13.75/2, 80, np.pi/2], [13.75/2, 130, np.pi/2]]) #cycle path

#path = np.array([[0, 0, 0], [50, 50, 0]]) #diagonal path
#path = np.array([[0, 0, 0], [50, 0, 0], [50, 50, np.pi/2], [0, 50, np.pi]])
test_path = np.array([[0, 0, 0], [50, 0, 99999], [50, 50, 99999], [0, 50, np.pi]])

#MPC
#mpc = MPC(current_path, mpc_horizon, mpc_dTime)
#mpc.objective(jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))


#error
angleErrors = []
xyErrors = []
xErrors = []
yErrors = []
angleVelErrors = []
xVelErrors = []
yVelErrors = []
def runIteration(newDt, localization_mode, add_errors):
    iter_count = 0
    #np.random.seed(0) 

    iterations = int(endTime/newDt)
    actual_robot = RobotActual(actual_dTime, newDt)
    ekf_robot = RobotEKF(newDt, localization_mode)
    
    xPID = PID(5, 0, 0)
    yPID = PID(5, 0, 0)
    thetaPID = PID(15, 0, 0)
    followRadius = 4
    #pp = PurePursuit(CYCLE_HUB, followRadius, xPID, yPID, thetaPID)
    pp = PurePursuit(ACCURACY_PATH, followRadius, xPID, yPID, thetaPID)

    currentTime = 0

    angleSum = 0.
    xSum = 0
    ySum = 0
    xySum = 0.
    angleVelSum = 0.
    xVelSum = 0.
    yVelSum = 0.
    paths_completed = 0
    total_paths = 10
    error_list = []
    #while(paths_completed < total_paths):
     #   while (not(pp.endedPath(ekf_robot.getPrevState()))):
    for _ in range(iterations):
            iter_count += 1
            #actual_robot.reset()
            #prev_state = ekf_robot.getPrevState()
            
            u = pp.step(ekf_robot.getPrevState()) #pure pursuit
            #u = mpc.step(prev_state) #model predictive control
            #u = np.array([[30, 0, 30, 0]]).T #circle
            #u = np.array([[20, 20, 20, 20]]).T #straight
            
            n = int(newDt/actual_dTime)
            # measurement = np.array([[0.], [0.], [0.], [0.], [0.], [0.]])
            # for _ in range(n):
            #     measurement += actual_robot.measure()
            #     actual_robot.step(u)
            # measurement /= n
            for _ in range(n):
                actual_robot.step(u)
            measurement = actual_robot.measure()
            ekf_robot.step(u, measurement)

            angleSum += ((actual_robot.getPrevState()[2] - ekf_robot.getPrevState()[2]) ** 2)/(iterations * 1.0)
            xySum += distance(actual_robot.getPrevState()[:2], ekf_robot.getPrevState()[:2])/(iterations * 1.0)
            xSum += ((actual_robot.getPrevState()[0] - ekf_robot.getPrevState()[0]) ** 2)/(iterations * 1.0)
            ySum += ((actual_robot.getPrevState()[1] - ekf_robot.getPrevState()[1]) ** 2)/(iterations * 1.0)

            angleVelSum += ((actual_robot.getPrevState()[5] - ekf_robot.getPrevState()[5]) ** 2)/(iterations * 1.0)
            xVelSum += ((actual_robot.getPrevState()[3] - ekf_robot.getPrevState()[3]) ** 2)/(iterations * 1.0)
            yVelSum += ((actual_robot.getPrevState()[4] - ekf_robot.getPrevState()[4]) ** 2)/(iterations * 1.0)
            
            currentTime += newDt
        #error_list.append(distance(actual_robot.getPrevState()[:2], ekf_robot.getPrevState()[:2]))
        #paths_completed += 1
        #if (paths_completed % 2 == 0):
        #    pp = PurePursuit(CYCLE_HUB, followRadius, xPID, yPID, thetaPID)
        #else:
        #    pp = PurePursuit(CYCLE_WAREHOUSE, followRadius, xPID, yPID, thetaPID)

    xSum = np.sqrt(xSum)
    ySum = np.sqrt(ySum)
    angleSum = np.sqrt(angleSum)
    xySum = np.sqrt(xySum)
    angleVelSum = np.sqrt(angleVelSum)
    xVelSum = np.sqrt(xVelSum)
    yVelSum = np.sqrt(yVelSum)

    if (add_errors):
        angleErrors.append(angleSum)
        xErrors.append(xSum)
        yErrors.append(ySum)
        xyErrors.append(xySum)
        angleVelErrors.append(angleVelSum)
        xVelErrors.append(xVelSum)
        yVelErrors.append(yVelSum)

    print("iter")
    return (ekf_robot.states, actual_robot.states, np.array(error_list))

#main loop
dtValues = []

avgXYError = []
avgXError = []
avgYError = []
avgAngleError = []
avgXVelError = []
avgYVelError = []
avgAngleVelError = []
mpc_dTime = 0.1

endTime = 15.0 #default = 15.0
actual_dTime = 0.001 #default = 0.001
startDt = 0.01

iters = 1
LOCALIZATION_MODE = 0 #0 is EKF, 1 is prediction, 2 is measurement
initial_state = np.array([0, 0, 0, 0, 0, 0])
#initial_state = np.array([13.75/2, 130, np.pi/2, 0, 0, 0])

#endDt = 5.00001
ACCURACY_PATH = np.array([[0, 0, np.pi/2], [0, 25, np.pi/6], [50, 50, np.pi/6], [50, 75, np.pi/2],
                                [75.3, 71.7, 99999], [87.4, 66.7, 99999], [95.2, 61.7, 99999],
                                [100.7, 56.7, 99999], [104.5, 51.7, 99999], [107.2, 46.7, 99999],
                                [108.7, 31.7, 99999], [107.2, 26.7, 99999], [104.5, 21.7, 99999],
                                [100.7, 16.7, 99999], [95.2, 11.7, 99999], [87.4, 6.7, 99999],
                                [75.3, 1.7, np.pi]])
current_path = ACCURACY_PATH

showFakeField = True
showField = True
showObstacles = True

endDt = 0.01
dtChange = 0.001
multiplier = 1.7
total_iter = 0
error_avg_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
while (startDt <= endDt):
    xErrors.clear()
    yErrors.clear()
    xyErrors.clear()
    angleErrors.clear()
    xVelErrors.clear()
    yVelErrors.clear()
    angleVelErrors.clear()

    dtValues.append(startDt)

    #for _ in range (iters):
   #     images = []
   #     (estimatedOdometryStates, actualOdometryStates, error_list) = runIteration(startDt, 0, True)
        #error_avg_list += error_list/20.0

    images = []
    (estimatedOdometryStates, actualOdometryStates, error_list) = runIteration(startDt, 0, True)
    print("error list:" + str(error_list))
    # (estimatedOdometryStates, actualOdometryStates) = runIteration(startDt, 0, True)
    # (estimatedModelStates, actualModelStates) = runIteration(startDt, 1, True)
    # (estimatedOnlyOdometryStates, actualOnlyOdometryStates) = runIteration(startDt, 2, True)

    images = [image1Sim, image2Sim]
    # (estimatedCameraStates, actualCameraStates) = runIteration(startDt, 1, True)
    (estimatedBothMeasurementsStates, actualBothMeasurementsStates, error_list) = runIteration(startDt, 0, True)

    avgXYError.append(sum(xyErrors)/(iters * 1.0))
    avgXError.append(sum(xErrors)/(iters * 1.0))
    avgYError.append(sum(yErrors)/(iters * 1.0))
    avgAngleError.append(sum(angleErrors)/(iters * 1.0))
    avgXVelError.append(sum(xVelErrors)/(iters * 1.0))
    avgYVelError.append(sum(yVelErrors)/(iters * 1.0))
    avgAngleVelError.append(sum(angleVelErrors)/(iters * 1.0))

    startDt += dtChange
    dtChange *= multiplier
startDt -= (dtChange/multiplier)
print("cycle error average:" + str(error_avg_list))
print("-------------------------------------")
print("AVERAGE RMSE VALUES:")
print("-------------------------------------")
print("XY ERROR: " + str(statistics.mean(avgXYError)))
print("X ERROR: " + str(statistics.mean(avgXError)))
print("Y ERROR: " + str(statistics.mean(avgYError)))
print("ANGLE ERROR: " + str(statistics.mean(avgAngleError)))
print("-------------------------------------")
print("X VEL ERROR: " + str(statistics.mean(avgXVelError)))
print("Y VEL ERROR: " + str(statistics.mean(avgYVelError)))
print("ANGLE VEL ERROR: " + str(statistics.mean(avgAngleVelError)))
print("-------------------------------------")
# print("ANGLE AVERAGES: " + str(averageAngleErrorList))
# print("ANGLE MEAN: " + str(statistics.mean(angleErrors)))
# #print("ANGLE STDEV: " + str(statistics.stdev(angleErrors)))
# print("-----------------------")
# print("TRANSLATIONAL AVERAGES: " + str(averageTransErrorList))
# print("TRANSLATIONAL MEAN: " + str(statistics.mean(translationalErrors)))
# #print("TRANSLATIONAL STDEV: " + str(statistics.stdev(translationalErrors)))

plt.figure(14)
plt.title("localization type comparison")
plt.xlabel("x position")
plt.ylabel("y position")
plt.plot(current_path[:, 0], current_path[:, 1])
#plt.plot(actualModelStates[:, 0], actualModelStates[:, 1])
#plt.plot(actualOnlyOdometryStates[:, 0], actualOnlyOdometryStates[:, 1])
plt.plot(actualOdometryStates[:, 0], actualOdometryStates[:, 1])
# plt.plot(actualCameraStates[:, 0], actualCameraStates[:, 1])
plt.plot(actualBothMeasurementsStates[:, 0], actualBothMeasurementsStates[:, 1])
if (showFakeField):
    plt.plot([-10, 130, 130, -10, -10], [-20, -20, 100, 100, -20], color = 'black')
    plt.plot([image1Sim[0][0], image1Sim[1][0]], [image1Sim[0][1], image1Sim[1][1]], color = '#00b900', linewidth = 3)
    plt.plot([image2Sim[0][0], image2Sim[1][0]], [image2Sim[0][1], image2Sim[1][1]], color = '#00b900', linewidth = 3)
plt.legend(['Target Path', 'odo + model', 'camera + odo + model'])
#plt.legend(['Target Path', 'model', 'odo', 'odo + model', 'camera + model', 'camera + odo + model'])


plt.figure(12)
plt.title("dt vs translation error")
plt.xlabel("dt")
plt.ylabel("RMSE translational error")
plt.xscale('log') 
plt.plot(dtValues, avgXYError)

plt.figure(22)
plt.xlabel("cycle")
plt.ylabel("distance error")
odo_errors = [6.2699843, 6.51979658,  8.75518228, 11.05853413, 11.46729051, 14.98201586, 11.33114132, 16.39483793, 14.78910938, 20.78359689]
cam_errors = np.array([0.84273461,  7.67439305,  1.15268751, 13.24965737,  1.38891814, 12.55375929, 1.24337586, 11.27671568,  1.76402486, 14.60445325])/2.0
plt.plot([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], odo_errors)
plt.plot([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], cam_errors)
plt.legend(['model + odo', 'camera + odo + model'])



# plt.figure(13)
# plt.title("accuracy test path")
# plt.xlabel("x position")
# plt.ylabel("y position")
# plt.plot(current_path[:, 0], current_path[:, 1])

#print("state length: " + str(len(state)))

#print("estimated states: " + str(state))

# BEGINNING COMMENT 
# print("graphing")
# plt.figure(1)
# plt.xlabel('time')
# plt.ylabel('vel')
# plt.plot(actual_dTime * np.array(range(int(endTime / actual_dTime))), actual_states[:, 4])
# plt.plot(startDt * np.array(range(iterations)), state[:, 4])
# plt.legend(['Actual Y Vel', 'Estimated Y Vel'])

# plt.figure(2)
# plt.title('x velocity vs time')
# plt.xlabel('time')
# plt.ylabel('vel')

# plt.plot(startDt * np.array(range(iterations + 1)), ekf_robot.predicted_states[:, 3])
# plt.plot(startDt * np.array(range(iterations + 1)), actual_robot.measured_states[:, 3])
# plt.plot(actual_dTime * np.array(range(int(endTime / actual_dTime))), actual_states[:, 3])
# plt.plot(startDt * np.array(range(iterations)), state[:, 3])
# #print("x vel measurements: " + str(actual_robot.measured_states[:, 3]))
# plt.legend(['prediction', 'measurement', 'actual', 'estimation'])

#HIHIHIHIHI
# iterations = int(endTime/startDt)
# plt.figure(3)
# plt.xlabel('time')
# plt.ylabel('theta vel')
# plt.plot(actual_dTime * np.array(range(int(endTime / actual_dTime))), actualModelStates[:, 5], color = '#ff7f0e')
# plt.plot(startDt * np.array(range(iterations)), estimatedOnlyOdometryStates[:, 5], color = '#1f77b4')
# plt.plot(startDt * np.array(range(iterations)), estimatedOdometryStates[:, 5], color = '#d62728')
# plt.plot(startDt * np.array(range(iterations)), estimatedModelStates[:, 5], color = '#2ca02c')
# plt.legend(['actual', 'odo', 'odo + model', 'model'])

plt.figure(5)
plt.xlabel('x position')
plt.ylabel('y position')
plt.axis('equal')
plt.plot(current_path[:, 0], current_path[:, 1])
# plt.plot(current_path[:, 0], current_path[:, 1], color = 'blue')
#plt.plot(actualOdometryStates[:, 0], actualOdometryStates[:, 1])
plt.plot(actualOdometryStates[:, 0], actualOdometryStates[:, 1])
#barrier width = 5.56,  normal gap = 13.7,  shared gap = 13.75, hub radius = 9
if (showField):
    plt.plot([0, 72, 72, 0, 0], [0, 0, 144, 144, 0], color = 'black')
    plt.plot([image1[0][0], image1[1][0]], [image1[0][1], image1[1][1]], color = '#00b900', linewidth = 3)
    plt.plot([image2[0][0], image2[1][0]], [image2[0][1], image2[1][1]], color = '#00b900', linewidth = 3)
if (showObstacles):
    hub = plt.Circle((48, 60), 9, fill = False, linewidth = 1.5)
    plt.gca().add_artist(hub)
    plt.plot([13.7, 13.7, 72, 72, 13.7], [96 - 5.56/2, 96 + 5.56/2, 96 + 5.56/2, 96 - 5.56/2, 96 - 5.56/2], color = 'black')
    plt.plot([48 - 5.56/2, 48 - 5.56/2, 48 + 5.56/2, 48 + 5.56/2, 48 - 5.56/2], [96 + 5.56/2, 144 - 13.75, 144 - 13.75, 96 + 5.56/2, 96 + 5.56/2], color = 'black')
plt.legend(['Target Path', 'Actual Position', 'Field', 'Images'])

# for i in range(len(state)):
#     if (i % int(0.25/startDt) == 0):
#         plt.arrow(state[i][0], state[i][1], 4*np.cos(state[i][2]), 4*np.sin(state[i][2]), head_width = 0.5, color = 'red', fill = True)
#         #print()
        
# plt.figure(6)
# plt.xlabel('x position')
# plt.ylabel('y position')
# plt.title('estimated vs actual position')

# #plt.plot(ekf_robot.predicted_states[:, 0], ekf_robot.predicted_states[:, 1])
# #plt.plot(actual_robot.measured_states[:, 0], actual_robot.measured_states[:, 1])
# plt.plot(actual_states[:, 0], actual_states[:, 1])
# plt.plot(state[:, 0], state[:, 1])
# plt.legend(['Actual Position', 'Estimated Position'])
# #plt.legend(['Predicted Position', 'Measured Position','Estimated Position', 'Actual Position'])

# plt.plot([0, 72, 72, 0, 0], [0, 0, 144, 144, 0])
# plt.plot()
# plt.plot()
# plt.plot([image1[0][0], image1[1][0]], [image1[0][1], image1[1][1]], color = 'red', linewidth = 2)
# plt.plot([image2[0][0], image2[1][0]], [image2[0][1], image2[1][1]], color = 'red', linewidth = 2)


# plt.figure(7)
# plt.axes().set_aspect('equal')
# plt.xlabel('x velocity')
# plt.ylabel('y velocity')

# plt.plot(actual_robot.measured_states[:, 3], actual_robot.measured_states[:, 4])
# plt.plot(actual_states[:, 3], actual_states[:, 4])
# plt.plot(state[:, 3], state[:, 4])
# plt.plot(ekf_robot.predicted_states[:, 3], ekf_robot.predicted_states[:, 4])

# plt.legend(['Measured Velocity', 'Actual Velocity', 'Estimated Velocity', 'Predicted Velocity'])

# plt.figure(8)
# plt.xlabel('time')
# plt.ylabel('encoder')
# plt.plot(dTime * np.array(range(iterations + 1)), ekf_robot.raw_measurements[:, 5])
# plt.plot(dTime * np.array(range(iterations + 1)), ekf_robot.predicted_measurements[:, 5])
# plt.legend(['Actual Measurement', 'Predicted Measurement'])
# END COMMENT

# plt.figure(7)
# plt.xlabel("time")
# plt.ylabel("covar trace")
# plt.plot(startDt * np.array(range(len(measured_var))), measured_var)
# plt.plot(startDt * np.array(range(len(measured_var))), predicted_var)
# plt.legend(['measurement', 'prediction'])
#plt.plot(x_axis, stats.norm.pdf(x_axis, measured_vels[250][0], measure_stddev))
#plt.legend(['Measurement', 'Prediction', 'Estimate'])

#plt.figure(8)
#plt.xlabel('xPos')
#plt.ylabel('yPos')
#plt.plot(np.array(predicted_trajectory)[:, 0], np.array(predicted_trajectory)[:, 1])
#plt.plot([0, 20], [0, 20])
#plt.legend(['optimized', 'target'])

#plt.figure(9)
#plt.xlabel('time')
#plt.ylabel('theta optimization')
#plt.plot(0.2 * np.array(range(len(predicted_trajectory))), np.array(predicted_trajectory)[:, 2])

# plt.figure(10)
# plt.xlabel('x pos')
# plt.ylabel('y pos')
# print(targets)
# plt.plot((np.array(targets))[:, 0], (np.array(targets))[:, 1])
# plt.legend(['Target Positions'])


#MPC optimization trajectory
# opt_x = []
# opt_y = []
# for i in range(len(MPC_trajectory)):
#     if (i % 6 == 0):
#         opt_x.append(MPC_trajectory[i])
#     if (i % 6 == 1):
#         opt_y.append(MPC_trajectory[i])
# plt.figure(11)
# plt.plot(opt_x, opt_y)
# plt.xlabel('x pos')
# plt.ylabel('y pos')
# plt.legend(['Optimization Positions'])

plt.show()


