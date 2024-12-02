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

    chassis_config = robot_state[:3]  # x, y, theta
    arm_config = robot_state[3:8]
    wheel_config = robot_state[8:]
    wheel_speeds = robot_speeds[:4]
    arm_speeds = robot_speeds[4:]

    # limit the speeds to the max allowed (negative and positive)
    if max_wheel_motor_speed:
        wheel_speeds = np.clip(
            wheel_speeds, -max_wheel_motor_speed, max_wheel_motor_speed
        )
    if max_arm_motor_speed:
        arm_speeds = np.clip(
            arm_speeds, -max_arm_motor_speed, max_arm_motor_speed
        )

    new_arm_config = arm_config + arm_speeds * dt
    new_wheel_config = wheel_config + wheel_speeds * dt

    # TODO: Odometry
    new_chassis_config = chassis_config

    new_robot_state = np.concatenate(
        [new_chassis_config, new_arm_config, new_wheel_config]
    )
    return new_robot_state


def simulate(
        initial_robot_state,
        constant_arm_speed,
        constant_wheel_speed,
        total_time
):
    dt = total_time // 100
    state = initial_robot_state
    states = [state]
    for t in range(dt):
        state = next_state(
            state,
            np.concatenate([constant_arm_speed, constant_wheel_speed]),
            dt
        )
        states.append(state)
    return states
