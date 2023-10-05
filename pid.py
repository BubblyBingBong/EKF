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

