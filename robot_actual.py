L_dis = 5
l_dis = 6.5
r_dis = 1
W_dis = 2
D_dis = 1


class RobotActual:
    def __init__(self, dt, measureDt):
        self.dt = dt
        self.measureDt = measureDt
        self.states = np.zeros([1, 6])  # np.array([[0., 0., 0., 0., 0., 0.]])
        # print("size: " + str(len(self.states)))
        self.measured_states = np.zeros([1, 6])
        self.firstUpdate = True
        self.firstMeasureUpdate = True

    def reset(self):
        self.states = np.zeros([1, 6])  # np.array([[0., 0., 0., 0., 0., 0.]])
        # print("size: " + str(len(self.states)))
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
        M = (1 / r_dis) * np.array(
            [
                [1, -1, -(L_dis + l_dis)],
                [1, 1, L_dis + l_dis],
                [1, 1, -(L_dis + l_dis)],
                [1, -1, L_dis + l_dis],
            ]
        )
        M_inv = np.linalg.pinv(M)
        M_inv_u = M_inv @ u
        A = M_inv_u[0][0]
        B = M_inv_u[1][0]
        C = M_inv_u[2][0]
        xVel = A
        yVel = B
        thetaVel = C
        Q = np.array(
            [
                [self.dt * 2e-3, 0, 0, 0, 0, 0],
                [0, self.dt * 2e-3, 0, 0, 0, 0],
                [0, 0, self.dt * 2e-3, 0, 0, 0],
                [0, 0, 0, 4.5e-1, 0, 0],
                [0, 0, 0, 0, 4.5e-1, 0],
                [0, 0, 0, 0, 0, 4.5e-1],
            ]
        )
        # Q = np.zeros((6, 6)) #MPC
        q = np.array(
            [
                [np.sqrt(Q[0][0]) * np.random.randn()],
                [np.sqrt(Q[1][1]) * np.random.randn()],
                [np.sqrt(Q[2][2]) * np.random.randn()],
                [np.sqrt(Q[3][3]) * np.random.randn()],
                [np.sqrt(Q[4][4]) * np.random.randn()],
                [np.sqrt(Q[5][5]) * np.random.randn()],
            ]
        )

        reality = [[xPos, yPos, theta, xVel, yVel, thetaVel]] + q.T
        if self.firstUpdate:
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

        local_xVel = (L + R) / 2
        local_thetaVel = (R - L) / (2 * W_dis)
        local_yVel = a + D_dis * local_thetaVel

        local_vel = np.array([[local_xVel, local_yVel, local_thetaVel]]).T

        return local_vel

    def measure(self):
        # odometry measurement
        actual_state = self.getPrevState()
        measurement = self.stateToMeasurement(actual_state)
        # measured_localVel = self.measurementToLocalVel(measurement)
        # measured_globalVel = rotationMatrix(actual_state[2]) @ measured_localVel
        # actual_measurements.append(measured_globalVel.T[0])
        theta = actual_state[2]
        xVel = actual_state[3]
        yVel = actual_state[4]
        thetaVel = actual_state[5]

        [bestImage, detected] = findBestImage(np.array([actual_state]).T)
        d = distance(actual_state, centerOfImage(bestImage))
        # print(measurement)

        R = np.array(
            [
                [1e-3 * d**2 + 1e-3, 0, 0, 0, 0, 0],
                [0, 1e-3 * d**2 + 1e-3, 0, 0, 0, 0],
                [0, 0, 1e-3 * d**2 + 1e-3, 0, 0, 0],
                [0, 0, 0, 2e-3 * measurement[3][0] ** 2 + 1e-3, 0, 0],
                [0, 0, 0, 0, 2e-3 * measurement[4][0] ** 2 + 1e-3, 0],
                [0, 0, 0, 0, 0, 2e-3 * measurement[5][0] ** 2 + 1e-3],
            ]
        )
        # R[3][3] = 0.
        # R[4][4] = 0.
        # R[5][5] = 0.
        # print(R[3])
        r = np.array(
            [
                [np.sqrt(R[0][0]) * np.random.randn()],
                [np.sqrt(R[1][1]) * np.random.randn()],
                [np.sqrt(R[2][2]) * np.random.randn()],
                [np.sqrt(R[3][3]) * np.random.randn()],
                [np.sqrt(R[4][4]) * np.random.randn()],
                [np.sqrt(R[5][5]) * np.random.randn()],
            ]
        )

        noisy_measurement = measurement + r
        # print(measurement.T)
        return noisy_measurement
        # prev_measurement = self.getPrevMeasurement()
        # localVel = self.measurementToLocalVel(noisy_measurement)
        # globalVel = rotationMatrix(theta) @ localVel
        # new_measurement = np.array([[prev_measurement[0] + self.measureDt*prev_measurement[3],
        #                              prev_measurement[1] + self.measureDt*prev_measurement[4],
        #                              prev_measurement[2] + self.measureDt*prev_measurement[5],
        #                              globalVel[0][0], globalVel[1][0], globalVel[2][0]]])
        # self.measured_states = np.append(self.measured_states, new_measurement, axis = 0)
        # print(self.measured_states)

    def getPrevState(self):
        if self.firstUpdate:
            return initial_state
        else:
            return self.states[-1]

    def getPrevMeasurement(self):
        if self.firstMeasureUpdate:
            self.firstMeasureUpdate = False
            return initial_state
        else:
            return self.measured_states[-1]
