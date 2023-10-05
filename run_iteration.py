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

