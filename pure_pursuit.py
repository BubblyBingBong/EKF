class PurePursuit:
    def __init__(self, path, followRadius, xPID, yPID, thetaPID):
        self.path = path
        self.followRadius = followRadius
        self.pathIndex = 1  # endpoint of current segment
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
            currentSegment = Segment(self.path[i - 1], self.path[i])
            if i == len(self.path) - 1:
                intersections = followCircle.getPathEndIntersections(currentSegment)
            else:
                intersections = followCircle.getSegmentIntersections(currentSegment)

            if len(intersections) > 0:
                targetPoint = currentSegment.closestToEnd(intersections)
                self.pathIndex = np.max([self.pathIndex, i])
                break
        if doublesEqual(targetPoint[0], 123456):
            targetPoint = self.path[self.pathIndex][:2]

        endPoint = self.path[-1][:2]
        if (
            not (self.followingEndPoint)
            and np.sum((endPoint[:2] - targetPoint[:2]) ** 2) < 0.01
        ):
            self.followingEndPoint = True

        targetPose = [targetPoint[0], targetPoint[1], self.path[self.pathIndex][2]]
        if self.followingEndPoint:
            # print("following end")
            targetPose = self.path[-1]
        # print("index: " + str(self.pathIndex))
        if abs(self.path[self.pathIndex][2] - 99999) < 0.1:
            # print("hello bello")
            targetPose[2] = math.atan2(
                targetPoint[1] - state[1], targetPoint[0] - state[0]
            )

        targets.append(targetPose)
        return targetPose

    def step(self, state):
        targetPose = self.getTargetPose(state)
        globalError = np.array(
            [
                [targetPose[0] - state[0]],
                [targetPose[1] - state[1]],
                [targetPose[2] - state[2]],
            ]
        )
        localError = rotationMatrix(-state[2]) @ globalError

        xPow = self.xPID.step(localError[0][0])
        yPow = self.yPID.step(localError[1][0])
        thetaPow = self.thetaPID.step(localError[2][0])
        m1 = xPow - yPow - thetaPow
        m2 = xPow + yPow + thetaPow
        m3 = xPow + yPow - thetaPow
        m4 = xPow - yPow + thetaPow
        u = np.array([[m1, m2, m3, m4]]).T
        scaleDown = max(30.0, max(abs(m1), max(abs(m2), max(abs(m3), abs(m4)))))
        # print(scaleDown)
        u *= 30.0 / scaleDown
        return u

    def endedPath(self, state):
        return distance(state, self.path[-1]) < 3
