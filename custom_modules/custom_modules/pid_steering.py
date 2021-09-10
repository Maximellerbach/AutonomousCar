import time


class SimpleSteering:
    """
    PD controller to steer the car
    """

    def __init__(self, kp=1, kd=1, high_th=1, low_th=-1):
        self.controller_settings = (kp, kd)
        self.ser = None

        self.high_th = high_th
        self.low_th = low_th

        # inputs
        self.lat_position = 0  # relative position to the center of the circuit
        self.last_received = time.time()

        # outputs
        self.steering = 0

        # some other stuff
        self.cte = 0
        self.cte_rate = 0

    def update_steering(self, lat_position, last_received=None):
        if last_received is None:
            last_received = time.time()

        cte = lat_position
        cte_rate = (cte - self.cte) / (last_received - self.last_received)

        self.lat_position = lat_position
        self.last_received = last_received

        self.cte = cte
        self.cte_rate = cte_rate

        self.steering = self.cte * self.controller_settings[0] + self.cte_rate * self.controller_settings[1]
        return self.steering
