import numpy as np
import matplotlib.pyplot as plt
import modern_robotics as mr
from robot_constants import RC


def odometry(chassis_config, delta_wheel_config):
    # The function Odometry is based on the equations in Chapter 13.4
    # It computes the new chassis configuration based on the old configuration, wheel speeds, and timestep
    # The output is a new chassis configuration

    # Get configs
    x, y, phi = chassis_config

    # delta_theta is the difference in wheel angles
    # Since we are assuming constant wheel speeds, dt = 1
    dt = 1  # Use actual timestep between wheel displacements for non-constant speeds
    theta_dot = delta_wheel_config / dt
    V_b = RC.F @ theta_dot
    V_b6 = np.array(0, 0, *V_b, 0)
    # Integrate to get the displacement: T_bb' = exp(V_b6)
    T_bb_prime = mr.MatrixExp6(V_b6)
    # If w_bz = 0, then delta_q_b = [0,v_bx,v_by]
    # Otherwise, delta_q_b = [w_bz, ...,...]
    w_bz = V_b[0]
    v_bx = V_b[1]
    v_by = V_b[2]
    if w_bz == 0:
        delta_q_b = np.array([0, v_bx, v_by])
    else:
        delta_q_b = np.array([
            w_bz,
            (v_bx*np.sin(w_bz) + v_by*(np.cos(w_bz) - 1)) / w_bz,
            (v_by*np.sin(w_bz) + v_bx*(1 - np.cos(w_bz))) / w_bz
        ])

    delta_q = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ]) @ delta_q_b

    return chassis_config + delta_q


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

    new_chassis_config = odometry(
        chassis_config=chassis_config,
        delta_wheel_config=np.subtract(new_wheel_config - wheel_config)
    )

    new_robot_state = np.concatenate(
        [new_chassis_config, new_arm_config, new_wheel_config]
    )
    return new_robot_state


def simulate(
        initial_robot_state,
        arm_speeds,
        wheel_speeds,
        total_time,
        dt=0.01
):
    N = total_time // dt
    print(f'Simulating for {total_time} seconds with {N} steps (dt={dt})')
    states = [initial_robot_state]
    for t in range(dt):
        current_state = states[-1]  # Latest state
        new_state = next_state(
            current_state,
            np.concatenate([arm_speeds, wheel_speeds]),
            dt
        )
        states.append(new_state)
        print(f'Step {t+1}/{N}') if t % 100 == 0 else None
    print(
        f'Finished simulation with {len(states)} states'
    )
    return states


def plot_states(states):
    # Plots the output of simulate
    # states is a list of robot states
    # where each robot state is a 12x1 vector
    # 3 for chassis config, 5 for arm, 4 for wheel angles
    states = np.array(states)
    # Chassis configs
    x = states[:, 0]
    y = states[:, 1]
    phi = states[:, 2]
    # Arm configs
    theta1 = states[:, 3]
    theta2 = states[:, 4]
    theta3 = states[:, 5]
    theta4 = states[:, 6]
    theta5 = states[:, 7]
    # Wheel configs
    wheel1 = states[:, 8]
    wheel2 = states[:, 9]
    wheel3 = states[:, 10]
    wheel4 = states[:, 11]

    # Plot the trajectory of the robot's chassis
    # Plot the orientation of the chassis as an arrow
    plt.figure()
    plt.plot(x, y)
    for i in range(0, len(x), 100):
        plt.arrow(
            x[i], y[i],
            0.1 * np.cos(phi[i]), 0.1 * np.sin(phi[i]),
            head_width=0.05, head_length=0.1, fc='r', ec='r'
        )
    plt.title('Chassis Trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.savefig('results/chassis_trajectory.png')

    # Plot the arm joint angles
    plt.figure()
    plt.plot(theta1, label='Joint 1')
    plt.plot(theta2, label='Joint 2')
    plt.plot(theta3, label='Joint 3')
    plt.plot(theta4, label='Joint 4')
    plt.plot(theta5, label='Joint 5')
    plt.title('Arm Joint Angles')
    plt.xlabel('Time Step')
    plt.ylabel('Angle [rad]')
    plt.legend()
    plt.savefig('results/arm_joint_angles.png')

    # Plot the wheel angles
    plt.figure()
    plt.plot(wheel1, label='Wheel 1')
    plt.plot(wheel2, label='Wheel 2')
    plt.plot(wheel3, label='Wheel 3')
    plt.plot(wheel4, label='Wheel 4')
    plt.title('Wheel Angles')
    plt.xlabel('Time Step')
    plt.ylabel('Angle [rad]')
    plt.legend()
    plt.savefig('results/wheel_angles.png')


def main():
    initial_robot_configuration = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])

    def rad_s(rpm):
        return (rpm * 2 * np.pi) / 60  # rad/s
    arm_speeds = np.array([0, 0, 0, 0, 0])  # rad/s
    wheel_rpm = 30
    wheel_speeds = np.array(
        [rad_s(wheel_rpm), rad_s(wheel_rpm),
         rad_s(wheel_rpm), rad_s(wheel_rpm)]
    )
    states = simulate(
        initial_robot_state=initial_robot_configuration,
        arm_speeds=arm_speeds,
        wheel_speeds=wheel_speeds,
        total_time=10
    )
    plot_states(states)


if __name__ == '__main__':
    main()
