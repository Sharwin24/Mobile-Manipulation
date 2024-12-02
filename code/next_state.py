import numpy as np


def next_state(
        robot_state,
        robot_speeds,
        dt,
        max_arm_motor_speed=None,
        max_wheel_motor_speed=None
):
    # robot_state is a 12x1 vector
    # 3 for chassis config, 5 for arm, 4 for wheel angles
    # robot_speeds is a 9x1 vector
    # 4 wheel speeds, 5 arm speeds
    # dt is the timestep
    # max_arm_motor_speed is a scalar that limits the arm motor speed
    # max_wheel_motor_speed is a scalar that limits the wheel motor speed

    # The function NextState is based on a simple first-order Euler step, i.e.,
    # new arm joint angles = (old arm joint angles) + (joint speeds) * dt
    # new wheel angles = (old wheel angles) + (wheel speeds) * dt
    # new chassis configuration is obtained from odometry, as described in Chapter 13.4

    # The output is a robot_state vector after the timestep
