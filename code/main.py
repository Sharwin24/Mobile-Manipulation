import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt

from robot_constants import RC
from next_state import next_state
from trajectory_generator import trajectory_generator, traj_to_sim_state, state_to_transform
from feedback_control import feedback_control, compute_robot_speeds

# Scene setup
# Initial Cube Configuration [x, y, theta] -> [1, 0, 0]
initial_cube_pose = [1, 0, 0]
# Final Cube Configuration [x, y, theta] -> [0, -1, -pi/2]
final_cube_pose = [0, -1, -np.pi/2]
initial_ee_config = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.5],
    [0, 0, 0, 1]
])
# EE Standoff Configuration (T_ce, standoff)
standoff_config = np.array([
    [0, 0, 1, -0.1],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.025 + 0.2],
    [0, 0, 0, 1]
])
# EE Grasping Configuration, relative to cube (T_ce, grasp)
ee_grasping_config = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1]
])

# Robot state is a 12x1 vector [chassis_config, arm_config, wheel_config]
# Chassis Config: [phi, x, y]
initial_chassis_config = np.array([0, 0, 0])
# Arm Config: [theta1, theta2, theta3, theta4, theta5]
initial_arm_config = np.array([0, 0, 0.2, -1.6, 0])
# Wheel Config: [thetaL1, thetaL2, thetaR1, thetaR2]
initial_wheel_config = np.array([0, 0, 0, 0])
# Gripper state: [0] for open, [1] for closed
initial_gripper_state = 0
initial_robot_state = np.concatenate(
    [initial_chassis_config, initial_arm_config,
     initial_wheel_config, [initial_gripper_state]]
)


def pose_to_transformation(x, y, theta):
    # Given an X,Y,theta in space frame, return the transformation matrix
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, x],
        [np.sin(theta), np.cos(theta), 0, y],
        [0, 0, 1, 0.025],
        [0, 0, 0, 1]
    ])


def create_trajectory(file: str = 'final_trajectory.csv'):
    T_cube_initial = pose_to_transformation(*initial_cube_pose)
    T_cube_final = pose_to_transformation(*final_cube_pose)
    # Generate trajectory
    traj, gripper_states = trajectory_generator(
        ee_initial_config=initial_ee_config,
        cube_initial_config=T_cube_initial,
        cube_final_config=T_cube_final,
        ee_grasping_config=ee_grasping_config,
        standoff_config=standoff_config,
        num_reference_configs=1
    )
    # Convert trajectory to simulation states and save to file
    sim_traj = traj_to_sim_state(
        trajectory=traj,
        gripper_states=gripper_states,
        write_to_file=True,
        filename=file
    )
    print(
        f'Trajectory saved to data/{file}, with {len(sim_traj)} states.'
    )
    return sim_traj


def plot_error(errors):
    # Plot the error over time
    errors = np.array(errors)
    # Error is a 6x1 vector [w_x, w_y, w_z, v_x, v_y, v_z]
    plt.figure()
    plt.plot(errors[:, 0], label='w_x')
    plt.plot(errors[:, 1], label='w_y')
    plt.plot(errors[:, 2], label='w_z')
    plt.plot(errors[:, 3], label='v_x')
    plt.plot(errors[:, 4], label='v_y')
    plt.plot(errors[:, 5], label='v_z')
    plt.legend()
    plt.title('Error Over Time')
    plt.xlabel('Sim Steps')
    plt.ylabel('Error')
    plt.grid()
    plt.savefig('results/error_over_time.png')


def plot_robot_states(states):
    # Plots the output of simulate
    # states is a list of robot states
    # where each robot state is a 12x1 vector
    # 3 for chassis config, 5 for arm, 4 for wheel angles
    # Chassis configs
    states = np.array(states)
    x = states[:, 1]
    y = states[:, 2]
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
    plt.plot(x[0], y[0], 'ro', label='Start')
    plt.plot(x[-1], y[-1], 'go', label='End')
    plt.title('Chassis Trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.savefig(f'results/chassis_trajectory.png')

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
    plt.savefig(f'results/arm_joint_angles.png')

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
    plt.savefig(f'results/wheel_angles.png')


def main():
    sim_traj = create_trajectory()
    # Loop through reference trajectories generated. If it has N reference configurations, it will have N-1 steps
    # so the Nth configuration is the reference trajectory Xd, and the N+1th configuration as Xd_next
    N = len(sim_traj)
    robot_states = [initial_robot_state]
    errors = []
    for i in range(N-1):
        # Each time through the loop, you
        # calculate the control law using FeedbackControl and generate th wheel and joint controls using Je
        # Send the controls, config, and timestep to next_state to calculate the new configuration
        # store every kth configuration for later animation (Note that the reference trajectory has k reference configs per 0.01s)
        # k=1 for simplicity
        # Store every kth X_err 6-vector, so you can later plot the evolution of the error over time.

        # Latest robot state
        current_robot_state = robot_states[-1]
        arm_config = current_robot_state[3:8]
        X = RC.T_se(
            phi=current_robot_state[0],
            x=current_robot_state[1],
            y=current_robot_state[2],
            arm_thetas=arm_config
        )
        Xd = state_to_transform(sim_traj[i])
        Xd_next = state_to_transform(sim_traj[i+1])
        V, X_err = feedback_control(
            X=X,
            Xd=Xd,
            Xd_next=Xd_next,
            Kp=np.eye(6),
            Ki=np.zeros((6, 6)),
            dt=0.01
        )
        errors.append(X_err)
        # robot speeds are 9x1 vector [wheel_speeds, arm_speeds]
        robot_speeds = compute_robot_speeds(V, arm_config)
        new_state = next_state(
            robot_state=robot_states[-1],
            robot_speeds=robot_speeds,
            dt=0.01,
            max_wheel_motor_speed=30.0,  # [rad/s]
            max_arm_motor_speed=15.0  # [rad/s]
        )
        # Add the gripper state to the new state
        gripper_state = sim_traj[i][-1]
        new_state = np.concatenate([new_state, [gripper_state]])
        robot_states.append(new_state)
    print(
        f'Finished simulation with {len(robot_states)} states'
    )
    # After the loop, write to a csv file of configurations. If the total time of the motion is 15 seconds, the csv file
    # should have 1500 (or 1501) lines, corresponding to 0.01s between each config.
    # Load the CSV file into Scene6 to see the results.
    np.savetxt('data/final_robot_states.csv', robot_states, delimiter=',')

    # Plot the error and robot states
    plot_error(errors)
    plot_robot_states(robot_states)


if __name__ == '__main__':
    main()
