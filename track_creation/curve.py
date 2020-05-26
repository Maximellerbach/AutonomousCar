import numpy as np

class Curve:
    def __init__(self, a, b, c, d):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.d = np.array(d)

    def Lerp(self, a, b, t):
        return a + (b - a) * t

    def QuadraticCurve(self, a, b, c, t):
        p0 = self.Lerp(a, b, t)
        p1 = self.Lerp(b, c, t)
        return self.Lerp(p0, p1, t)

    def CubicCurve(self, t):
        p0 = self.QuadraticCurve(self.a, self.b, self.c, t)
        p1 = self.QuadraticCurve(self.b, self.c, self.d, t)
        return self.Lerp(p0, p1, t)

    def step_loop(self, start, end, it_step):
        current_it = start
        round_res = len(str(it_step))-2

        points = []
        while(current_it<end):
            current_it = round(current_it + it_step, round_res)
            pt = self.CubicCurve(current_it)
            points.append(pt)

        return np.array(points)

    def plot(self, start=0, end=1, it_step=0.1):
        import matplotlib.pyplot as plt

        points = self.step_loop(start, end, it_step)
        plt.scatter(points[:, 0], points[:, 1])
        plt.show()
        


if __name__ == "__main__":
    c = Curve([0, 0], [0, 1], [1, 1], [1, 2])
    c.plot(start=0, end=1, it_step=0.05)
    