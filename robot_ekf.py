
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

