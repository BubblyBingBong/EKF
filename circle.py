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

