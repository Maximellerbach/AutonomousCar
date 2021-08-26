import xbox_controller


if __name__ == '__main__':
    joy = xbox_controller.XboxController()
    while True:
        print(joy.read())
