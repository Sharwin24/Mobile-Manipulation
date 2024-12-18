import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from robot_constants import RC
from next_state import next_state
from trajectory_generator import trajectory_generator, traj_to_sim_state, state_to_transform, state_to_transform
from feedback_control import feedback_control, compute_robot_speeds, test_joint_limits

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
initial_chassis_config = np.array([np.deg2rad(30), 0.1, 0.2])
# Arm Config: [theta1, theta2, theta3, theta4, theta5]
initial_arm_config = np.array([0, -0.587, -0.9, 0, 0])
assert all(test_joint_limits(initial_arm_config)
           ), 'Initial arm config is out of joint limits'
# Wheel Config: [thetaL1, thetaL2, thetaR1, thetaR2]
initial_wheel_config = np.array([0, 0, 0, 0])
# Gripper state: [0] for open, [1] for closed
initial_gripper_state = 0
initial_robot_state = np.concatenate(
    [initial_chassis_config, initial_arm_config,
     initial_wheel_config, [initial_gripper_state]]
)

# NewTask Scene Setup
initial_cube_pose_newTask = [1, 0.2, 0.4]  # [x, y, theta]
final_cube_pose_newTask = [0, -1, -2]  # [x, y, theta]


def pose_to_transformation(x, y, theta):
    # Given an X,Y,theta in space frame, return the transformation matrix
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, x],
        [np.sin(theta), np.cos(theta), 0, y],
        [0, 0, 1, 0.025],
        [0, 0, 0, 1]
    ])


def plot_error(errors, sim_name: str):
    # Convert errors to a numpy array
    errors = np.array(errors)

    # Create a figure with two subplots
    plt.figure(figsize=(12, 8))

    # Full graph of error over time
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title('Error Over Time')
    ax1.plot(errors[:, 0], label='w_x', color='r')
    ax1.plot(errors[:, 1], label='w_y', color='g')
    ax1.plot(errors[:, 2], label='w_z', color='b')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Sim Steps')
    ax1.set_ylabel('Angular Velocity Error [rad/s]')
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(errors[:, 3], label='v_x', color='r', linestyle='--')
    ax2.plot(errors[:, 4], label='v_y', color='g', linestyle='--')
    ax2.plot(errors[:, 5], label='v_z', color='b', linestyle='--')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Linear Velocity Error [m/s]')

    # Zoomed in graph of error over time
    zoom_steps = int(len(errors) * 0.15)  # Some % of the steps
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(errors[:zoom_steps, 0], label='w_x', color='r')
    ax3.plot(errors[:zoom_steps, 1], label='w_y', color='g')
    ax3.plot(errors[:zoom_steps, 2], label='w_z', color='b')
    ax3.legend(loc='upper left')
    ax3.set_title(f'Error Over Time (Zoomed In on first {
                  zoom_steps / len(errors) * 100:.1f}%)')
    ax3.set_xlabel('Sim Steps')
    ax3.set_ylabel('Angular Velocity Error [rad/s]')
    ax3.grid()

    ax4 = ax3.twinx()
    ax4.plot(errors[:zoom_steps, 3],
             label='v_x', color='r', linestyle='--')
    ax4.plot(errors[:zoom_steps, 4],
             label='v_y', color='g', linestyle='--')
    ax4.plot(errors[:zoom_steps, 5],
             label='v_z', color='b', linestyle='--')
    ax4.legend(loc='upper right')
    ax4.set_ylabel('Linear Velocity Error [m/s]')

    plt.tight_layout()
    plt.savefig(f'results/{sim_name}/error_plot.png')
    print(f'Combined error plot saved to results/{sim_name}/error_plot.png')

    # Also save errors to a CSV file
    np.savetxt(f'results/{sim_name}/errors.csv', errors, delimiter=',')


def plot_robot_states(states, reference_states, sim_name: str):
    # Plots the output of simulate
    # states is a list of robot states
    # where each robot state is a 12x1 vector
    # 3 for chassis config, 5 for arm, 4 for wheel angles

    # Chassis configs
    states = np.array(states)
    phi = states[:, 0]
    x = states[:, 1]
    y = states[:, 2]
    # Arm configs
    theta1 = states[:, 3]
    theta2 = states[:, 4]
    theta3 = states[:, 5]
    theta4 = states[:, 6]
    theta5 = states[:, 7]
    arm_configs = states[:, 3:8]
    # Wheel configs
    wheel1 = states[:, 8]
    wheel2 = states[:, 9]
    wheel3 = states[:, 10]
    wheel4 = states[:, 11]

    # Get joint limits for each joint and plot straight line
    min_limits = [jl[0] for jl in RC.joint_limits]
    max_limits = [jl[1] for jl in RC.joint_limits]
    # min_limits = np.full((N, 5), min_limits)
    # max_limits = np.full((N, 5), max_limits)

    # Actual T_se transformations
    actual_transformations = np.array(
        [RC.T_se(phi, x, y, arm_config)
         for phi, x, y, arm_config in zip(phi, x, y, arm_configs)]
    )
    actual_ee_x = actual_transformations[:, 0, 3]
    actual_ee_y = actual_transformations[:, 1, 3]

    # Reference T_se transformations
    reference_states = np.array(
        [state_to_transform(s) for s in reference_states]
    )
    reference_ee_x = reference_states[:, 0, 3]
    reference_ee_y = reference_states[:, 1, 3]

    # Create a single figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot the trajectory of the robot's chassis
    axs[0].plot(x, y, 'b-')
    axs[0].plot(x[0], y[0], 'go', label='Start')
    axs[0].plot(x[-1], y[-1], 'ro', label='End')
    # Also plot the reference trajectory in black dashed lines
    axs[0].plot(reference_ee_x, reference_ee_y, 'k:', label='EE Reference')
    # Plot the end effector position in green dotted lines
    axs[0].plot(actual_ee_x, actual_ee_y, 'b--', label='EE Actual')
    axs[0].set_title(f'Chassis Trajectory ({sim_name})')
    axs[0].set_xlabel('x [m]')
    axs[0].set_ylabel('y [m]')
    axs[0].legend()
    axs[0].grid()

    # Plot the arm joint angles
    jc = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'c'}
    axs[1].plot(theta1, jc[1], label='Joint 1')
    axs[1].plot(theta2, jc[2], label='Joint 2')
    axs[1].plot(theta3, jc[3], label='Joint 3')
    axs[1].plot(theta4, jc[4], label='Joint 4')
    axs[1].plot(theta5, jc[5], label='Joint 5')
    # Plot joint limits
    for j in range(5):
        axs[1].axhline(
            min_limits[j],
            color=f'{jc[j+1]}',
            linestyle='--',
            label=f'Joint {j+1} Min'
        )
        axs[1].axhline(
            max_limits[j], color=f'{jc[j+1]}',
            linestyle='--',
            label=f'Joint {j+1} Max'
        )
    axs[1].set_title(f'Arm Joint Angles ({sim_name})')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Angle [rad]')
    axs[1].legend()

    # Plot the wheel angles
    axs[2].plot(wheel1, label='Wheel 1')
    axs[2].plot(wheel2, label='Wheel 2')
    axs[2].plot(wheel3, label='Wheel 3')
    axs[2].plot(wheel4, label='Wheel 4')
    axs[2].set_title(f'Wheel Angles ({sim_name})')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Angle [rad]')
    axs[2].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'results/{sim_name}/robot_states.png')
    print(f'Robot states plot saved to results/{sim_name}/robot_states.png')


# control_type options: ['FF+PI', 'P', 'PI']
sim_control_params = {
    "best": {'Kp': np.eye(6) * 2.0, 'Ki': np.eye(6) * 0.01, 'control_type': 'FF+PI'},
    "overshoot": {'Kp': np.eye(6) * 5.0, 'Ki': np.zeros((6, 6)), 'control_type': 'P'},
    "newTask": {'Kp': np.eye(6) * 5.0, 'Ki': np.eye(6) * 0.1, 'control_type': 'FF+PI'}
}


def create_new_state(
        state,
        V,
        arm_config,
        apply_joint_limits=False
):
    # robot speeds are 9x1 vector [wheel_speeds, arm_speeds]
    robot_speeds = compute_robot_speeds(V, arm_config, [])
    # Compute the next state using the current state and robot speeds
    new_state = next_state(
        robot_state=state,
        robot_speeds=robot_speeds,
        dt=0.01,
        max_wheel_motor_speed=RC.max_wheel_motor_speed,  # [rad/s]
        max_arm_motor_speed=RC.max_arm_motor_speed  # [rad/s]
    )
    if not apply_joint_limits:
        return new_state

    # Apply joint limits
    new_arm_state = new_state[3:8]
    joints_within_limits = test_joint_limits(new_arm_state)
    if not any(joints_within_limits):
        # Create a list of violated joints by joint number (index+1)
        violated_joints = [i for i, v in enumerate(
            joints_within_limits) if not v]
        # Recalculate the robot speeds with violated joints set to 0
        robot_speeds = compute_robot_speeds(V, arm_config, violated_joints)
        # Recalculate the new state with the new robot speeds
        limited_new_state = next_state(
            robot_state=state,
            robot_speeds=robot_speeds,
            dt=0.01,
            max_wheel_motor_speed=RC.max_wheel_motor_speed,  # [rad/s]
            max_arm_motor_speed=RC.max_arm_motor_speed  # [rad/s]
        )
        return limited_new_state
    return new_state


def main(sim_name: str):
    # Save sys.stdout to a log file
    # Create the file and directory if it doesn't exist already
    os.makedirs(f'results/{sim_name}', exist_ok=True)
    print(f'Starting simulation: {sim_name}')
    log_file = open(f'results/{sim_name}/{sim_name}_log.txt', 'w')
    sys.stdout = log_file
    print(f'Starting simulation: {sim_name}')
    if sim_name == 'newTask':
        T_cube_initial = pose_to_transformation(*initial_cube_pose_newTask)
        T_cube_final = pose_to_transformation(*final_cube_pose_newTask)
    else:
        T_cube_initial = pose_to_transformation(*initial_cube_pose)
        T_cube_final = pose_to_transformation(*final_cube_pose)
    # Generate trajectory
    traj, gripper_states = trajectory_generator(
        ee_initial_config=initial_ee_config,
        cube_initial_config=T_cube_initial,
        cube_final_config=T_cube_final,
        ee_grasping_config=ee_grasping_config,
        standoff_config=standoff_config,
        num_reference_configs=1,
        debug=False
    )
    # Convert trajectory to simulation states and save to file
    sim_traj = traj_to_sim_state(
        trajectory=traj,
        gripper_states=gripper_states,
        write_to_file=True,
        filename=f'{sim_name}/sim_trajectory.csv'
    )
    # Loop through reference trajectories generated. If it has N reference configurations, it will have N-1 steps
    # so the Nth configuration is the reference trajectory Xd, and the N+1th configuration as Xd_next
    N = len(sim_traj)
    robot_states = [initial_robot_state]
    errors = []
    Kp = sim_control_params[sim_name]['Kp']
    Ki = sim_control_params[sim_name]['Ki']
    for i in range(N-1):
        # Latest robot state
        current_robot_state = robot_states[-1]
        arm_config = current_robot_state[3:8]
        # Current robot configuration
        X = RC.T_se(
            phi=current_robot_state[0],
            x=current_robot_state[1],
            y=current_robot_state[2],
            arm_thetas=arm_config
        )
        # Using simulation state, convert to transformation matrix
        Xd = state_to_transform(sim_traj[i])
        Xd_next = state_to_transform(sim_traj[i+1])
        # Compute feedback control and save the error
        control_type = sim_control_params[sim_name]['control_type']
        V, X_err = feedback_control(
            X, Xd, Xd_next, Kp, Ki, control_type, dt=0.01
        )
        errors.append(X_err)
        new_state = create_new_state(
            current_robot_state, V, arm_config, apply_joint_limits=True
        )
        # Add the gripper state to the new state
        gripper_state = sim_traj[i][-1]
        new_state = np.concatenate([new_state, [gripper_state]])
        robot_states.append(new_state)
        if i % (N // 8) == 0:
            log_file.close()
            sys.stdout = sys.__stdout__
            print(f'{i/N * 100:.1f}%')
            log_file = open(f'results/{sim_name}/{sim_name}_log.txt', 'a')
            sys.stdout = log_file
    print(
        f'Finished simulation with {len(robot_states)} states'
    )
    # After the loop, write to a csv file of configurations. If the total time of the motion is 15 seconds, the csv file
    # should have 1500 (or 1501) lines, corresponding to 0.01s between each config.
    # Load the CSV file into Scene6 to see the results.
    np.savetxt(
        f'results/{sim_name}/robot_states.csv', robot_states, delimiter=','
    )

    # Plot the error and robot states
    plot_error(errors, sim_name)
    plot_robot_states(robot_states, sim_traj, sim_name)

    # Close the log file
    log_file.close()
    sys.stdout = sys.__stdout__
    print(f'Finished simulation: {sim_name}')


if __name__ == '__main__':
    main(sim_name='best')
    main(sim_name='overshoot')
    main(sim_name='newTask')
