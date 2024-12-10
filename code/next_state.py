import numpy as np
import matplotlib.pyplot as plt
import modern_robotics as mr
from robot_constants import RC
import os


def odometry(chassis_config, delta_wheel_config):
    # The function Odometry is based on the equations in Chapter 13.4
    # It computes the new chassis configuration based on the old configuration and the change in wheel angles

    # Get configs
    phi, x, y = chassis_config
    # delta_theta is the difference in wheel angles
    # Since we are assuming constant wheel speeds, dt = 1
    dt = 1  # Use actual timestep between wheel displacements for non-constant speeds
    theta_dot = delta_wheel_config / dt
    # Calculate the Body twist using the pinv(H0) and theta_dot
    V_b = RC.F @ theta_dot
    # Integrate to get the displacement: T_bk = exp([V_b6])
    V_b6 = np.array([0, 0, *V_b, 0])
    T_bk = mr.MatrixExp6(mr.VecTose3(V_b6))
    T_sk = RC.T_sb(phi, x, y) @ T_bk
    new_phi = np.arctan2(T_sk[1, 0], T_sk[0, 0])
    # new_phi = np.arccos(T_sk[0, 0])
    new_chassis_config = np.array([
        new_phi,
        T_sk[0, 3],
        T_sk[1, 3]
    ])
    return new_chassis_config


def next_state(
        robot_state,
        robot_speeds,
        dt,
        max_arm_motor_speed,
        max_wheel_motor_speed
):
    # robot_state is a 13x1 vector
    # 3 for chassis config, 5 for arm, 4 for wheel angles, 1 for gripper state
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

    chassis_config = robot_state[:3]  # phi, x, y
    arm_config = robot_state[3:8]
    wheel_config = robot_state[8:12]
    wheel_speeds = robot_speeds[:4]
    arm_speeds = robot_speeds[4:]

    # limit the speeds to the max allowed (negative and positive)
    wheel_speeds = np.clip(
        wheel_speeds, -max_wheel_motor_speed, max_wheel_motor_speed
    )
    arm_speeds = np.clip(
        arm_speeds, -max_arm_motor_speed, max_arm_motor_speed
    )

    new_arm_config = arm_config + (arm_speeds * dt)
    new_wheel_config = wheel_config + (wheel_speeds * dt)

    new_chassis_config = odometry(
        chassis_config=chassis_config,
        delta_wheel_config=(new_wheel_config - wheel_config)
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
    N = int(total_time / dt)
    print(f'Simulating for {total_time} seconds with {N} steps (dt={dt})')
    states = [initial_robot_state]
    for t in range(N):
        current_state = states[-1]  # Latest state
        new_state = next_state(
            current_state,
            np.concatenate([arm_speeds, wheel_speeds]),
            dt
        )
        states.append(new_state)
        # print(f'Step {t}/{N}') if t % 100 == 0 else None
    print(
        f'Finished simulation with {len(states)} states'
    )
    return states


def plot_states(states, sim_name):
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

    # Create the directories if they don't exist
    os.makedirs(f'results/{sim_name}', exist_ok=True)

    # Plot the trajectory of the robot's chassis
    # Plot the orientation of the chassis as an arrow
    plt.figure()
    plt.plot(x, y)
    plt.title('Chassis Trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.savefig(f'results/{sim_name}/chassis_trajectory.png')

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
    plt.savefig(f'results/{sim_name}/arm_joint_angles.png')

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
    plt.savefig(f'results/{sim_name}/wheel_angles.png')


def run_simulation(robot_initial_state, arm_speeds, wheel_speeds, total_time, sim_name):
    states = simulate(
        initial_robot_state=robot_initial_state,
        arm_speeds=arm_speeds,
        wheel_speeds=wheel_speeds,
        total_time=total_time
    )
    final_state = states[-1]
    print(
        f'{sim_name}: {
            np.round(robot_initial_state[:3], 3)} -> {np.round(final_state[:3], 3)}'
    )
    plot_states(states, sim_name)


def main():
    robot_initial_state = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    sim_names = ['straight', 'sideways', 'spin']
    wheel_speeds = np.array([
        [10, 10, 10, 10],
        [-10, 10, -10, 10],
        [-10, 10, 10, -10],
    ])
    # Arm speeds are 5 zeros with the same length as wheel_speeds
    arm_speeds = np.array([
        [0, 0, 0, 0, 0] for _ in range(len(wheel_speeds))
    ])
    for sim_name, arm_speed, wheel_speed in zip(sim_names, arm_speeds, wheel_speeds):
        run_simulation(
            robot_initial_state=robot_initial_state,
            arm_speeds=arm_speed,
            wheel_speeds=wheel_speed,
            total_time=1,
            sim_name=sim_name
        )


if __name__ == '__main__':
    main()
