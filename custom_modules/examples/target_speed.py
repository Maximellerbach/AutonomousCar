import time

from custom_modules import pid_controller

if __name__ == "__main__":
    controller = pid_controller.PIDController()
    controller.update_target(1)
    # controller.init_ser()

    state = 0
    while(True):
        new_pwm = controller.update(state, time.time())
        if new_pwm is not None:
            print(new_pwm)

        time.sleep(0.05)
