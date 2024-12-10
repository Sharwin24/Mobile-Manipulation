import modern_robotics as mr
import numpy as np

from robot_constants import RC
from next_state import simulate
from trajectory_generator import trajectory_generator, traj_to_sim_state

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


def run_simulation(sim_traj):
    # Loop through reference trajectories generated. If it has N reference configurations, it will have N-1 steps
    # so the Nth configuration is the reference trajectory Xd, and the N+1th configuration as Xd_next to
    # calculate the feedforward twist Vd

    # Each time through the loop, you
    # calculate the control law using FeedbackControl and generate th wheel and joint controls using Je
    # Send the controls, config, and timestep to next_state to calculate the new configuration
    # store every kth configuration for later animation (Note that the reference trajectory has k reference configs per 0.01s)
    # k=1 for simplicity
    # Store every kth X_err 6-vector, so you can later plot the evolution of the error over time.

    # After the loop, write to a csv file of configurations. If the total time of the motion is 15 seconds, the csv file
    # should have 1500 (or 1501) lines, corresponding to 0.01s between each config.
    # Load the CSV file into Scene6 to see the results.
    pass


def main():
    traj = create_trajectory()


if __name__ == '__main__':
    main()
