class Segment:
    def __init__(self, startPoint, endPoint):
        self.startPoint = startPoint
        self.endPoint = endPoint

    def isVertical(self):
        return doublesEqual(self.startPoint[0], self.endPoint[0])

    def getAngle(self):
        radians = 0
        if self.isVertical():
            if self.endPoint[1] > self.startPoint[1]:
                return np.pi / 2
            else:
                return -np.pi / 2
        else:
            return math.atan2(
                self.endPoint[1] - self.startPoint[1],
                self.endPoint[0] - self.startPoint[0],
            )

    def moveAlongSegment(self, distance):
        angle = self.getAngle()
        return [
            self.startPoint[0] + distance * np.cos(angle),
            self.startPoint[1] + distance * np.sin(angle),
        ]

    def function(self, x):
        m = (self.endPoint[1] - self.startPoint[1]) / (
            self.endPoint[0] - self.startPoint[0]
        )
        return m * (x - self.startPoint[0]) + self.startPoint[1]

    def inSegmentRange(self, point):
        return inBetween(point[0], self.startPoint[0], self.endPoint[0]) and inBetween(
            point[1], self.startPoint[1], self.endPoint[1]
        )

    def inRayRange(self, point):
        if self.endPoint[0] > self.startPoint[0]:
            withinX = point[0] >= self.startPoint[0]
        else:
            withinX = point[0] <= self.startPoint[0]

        if self.endPoint[1] > self.startPoint[1]:
            withinY = point[1] >= self.startPoint[1]
        else:
            withinY = point[1] <= self.startPoint[1]
        return withinX and withinY

    def closestToEnd(self, points):
        if len(points) == 0:
            return [0, 0]
        minDistance = 999999999
        closestPoint = points[0]
        for point in points:
            if distance(point, self.endPoint) < minDistance:
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
        if self.isVertical() and line.isVertical():
            xVal = 0
            yVal = 0
        elif not (self.isVertical()) and line.isVertical():
            xVal = line.startPoint[0]
            yVal = (self_y2 - self_y1) / (self_x2 - self_x1) * (
                xVal - self_x1
            ) + self_y1
        elif self.isVertical() and not (line.isVertical()):
            xVal = self.startPoint[0]
            yVal = (other_y2 - other_y1) / (other_x2 - other_x1) * (
                xVal - other_x1
            ) + other_y1
        else:  # both not vertical
            self_slope = (self_y2 - self_y1) / (self_x2 - self_x1)
            other_slope = (other_y2 - other_y1) / (other_x2 - other_x1)
            if doublesEqual(self_slope, other_slope):
                return [0, 0]
            self_yint = -self_slope * self_x1 + self_y1
            other_yint = -other_slope * other_x1 + other_y1

            xVal = (other_yint - self_yint) / (self_slope - other_slope)
            yVal = self_slope * xVal + self_yint
        return [xVal, yVal]

    def projectToLine(self, point):
        if self.isVertical():
            return [self.startPoint[0], point[1]]
        slope = (self.endPoint[1] - self.startPoint[1]) / (
            self.endPoint[0] - self.startPoint[0]
        )
        if doublesEqual(slope, 0):
            return [point[0], self.startPoint[1]]
        perp_slope = -1 / slope
        perp_line = Segment(point, [point[0] + 1, point[1] + perp_slope])
        return self.lineIntersection(perp_line)

    def projectToSegment(self, point):
        lineProjection = self.projectToLine(point)
        segmentProjection = lineProjection

        if (
            lineProjection[0] < min(self.startPoint[0], self.endPoint[0])
            or lineProjection[0] > max(self.startPoint[0], self.endPoint[0])
            or lineProjection[1] < min(self.startPoint[1], self.endPoint[1])
            or lineProjection[1] > max(self.startPoint[1], self.endPoint[1])
        ):
            if distance(lineProjection, self.startPoint) < distance(
                lineProjection, self.endPoint
            ):
                segmentProjection = self.startPoint
            else:
                segmentProjection = self.endPoint
        return segmentProjection
