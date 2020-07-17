import SerialCommand
import numpy as np
import time
import matplotlib.pyplot as plt


class KalmanFilter():
    def __init__(self, sensor_class, initial_state):
        self.mea_error = sensor_class.MEA_ERROR
        self.estimation = initial_state
        self.est_error = 1

    def update(self, new_state):
        new_state += np.random.normal()*self.mea_error
        
        self.kalman_gain = self.est_error/(self.est_error+self.mea_error)
        self.estimation = self.estimation + self.kalman_gain * (new_state-self.estimation)
        self.est_error = (1 - self.kalman_gain) * self.est_error
        if self.kalman_gain < 0.1:
            self.kalman_gain = 0.1

        return self.estimation, self.est_error
    
if __name__ == "__main__":
    KF = KalmanFilter(SerialCommand.compteTour, 0)

    new_state = 1
    states = []
    pred_states = []
    it = 0
    for i in range(1000):
        to_pred = new_state+np.random.normal()*SerialCommand.compteTour.MEA_ERROR
        est, error = KF.update(to_pred)
        print(new_state, est)

        states.append(to_pred)
        pred_states.append(est)

        it += 1

    plt.plot([i for i in range(len(states))], states)
    plt.plot([i for i in range(len(states))], pred_states)
    plt.show()