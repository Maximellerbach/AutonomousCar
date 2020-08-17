import SerialCommand
import numpy as np
import time
import matplotlib.pyplot as plt
import pykalman
import sensors

if __name__ == "__main__":
    sensor_list = [sensors.sensor_compteTour()]

    KF = pykalman.KalmanFilter(initial_state_mean=[i.INITIAL_STATE for i in sensor_list],
                               transition_covariance=[
                                   i.MEA_ERROR**2 for i in sensor_list],
                               observation_covariance=[i.MEA_ERROR for i in sensor_list])

    states = []
    to_preds = []
    pred_states = []
    abs_error = []
    abs_pred_error = []

    it = 0
    covariance = 0
    new_state = 0
    estimation = new_state

    for i in range(1000):
        new_state = np.sin(np.deg2rad(it % 360))
        to_pred = new_state+np.random.normal()*0.05

        estimation, covariance = KF.filter_update(
            estimation, covariance, to_pred)
        # print(new_state, estimation, covariance)

        states.append(new_state)
        to_preds.append(to_pred)
        pred_states.append(estimation[0][0])
        abs_error.append(abs(to_pred-new_state))
        abs_pred_error.append(abs(estimation[0][0]-new_state))

        it += 0.5

    plt.plot([i for i in range(len(states))], states)
    plt.plot([i for i in range(len(states))], to_preds)
    plt.plot([i for i in range(len(states))], pred_states)
    plt.plot([i for i in range(len(states))], abs_error)
    plt.plot([i for i in range(len(states))], abs_pred_error)
    plt.show()
