import sys
import time

from custom_modules import serial_command, rotatecar

if __name__ == "__main__":
    def getParams(argv):
        for it, arg in enumerate(argv):
            if arg in ("-a", "--angle"):
                angle = int(argv[it+1].strip())
            elif arg in ("-w", "--way"):
                way = int(argv[it+1].strip())
            elif arg in ("-o", "--orientation"):
                orientation = int(argv[it+1].strip())

        return angle, way, orientation

    # r = get_approx_radius(38)
    # d = distance_needed_to_turn(90, r)
    # d_remaining = remaining_distance(0.5, d)
    # print(r d, d_remaining)

    angle, way, orientation = getParams(sys.argv[1:])

    ser = serial_command.start_serial()
    rotatecar(ser, angle, way, orientation=orientation)

    time.sleep(1)
    ser.stop()
