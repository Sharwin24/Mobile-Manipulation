import modern_robotics as mr
import numpy as np


def traj_to_sim_state(trajectory, gripper_states, write_to_file=False, filename='trajectory.csv'):
    # Trajectory shape: (num_reference_configs, 4, 4)
    # Trajectory is a list of 4x4 transformation matrices that
    # represent the end-effector's configuration at each reference configuration
    # gripper_states shape: (num_reference_configs) where each element is a binary value [0, 1]
    # We need to convert that data into the following format:
    # r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state
    # where rij are the elements of the rotation matrix within the 4x4 transformation matrix
    # and px, py, pz are the position elements of the 4x4 transformation matrix
    # gripper state is a binary value representing the state of the gripper (0 = open, 1 = closed)
    sim_state = []
    for i, config in enumerate(trajectory):
        r = config[:3, :3]
        p = config[:3, 3]
        gripper_state = gripper_states[i]
        sim_state.append(np.concatenate([r.flatten(), p, [gripper_state]]))
    sim_state = np.array(sim_state)
    assert sim_state.shape == (
        len(trajectory), 13), f'Invalid simulation state shape {sim_state.shape}'
    if write_to_file:
        np.savetxt(filename, sim_state, delimiter=',')
        print(f'Successfully wrote simulation state to {filename}')
    return sim_state


def trajectory_generator(
        ee_initial_config,
        cube_initial_config,
        cube_final_config,
        ee_grasping_config,
        standoff_config,
        num_reference_configs: int,
        debug: bool = False
):
    """
      Generate a reference trajectory for the end-effector frame. This trajectory consists of 8 concatenated
      trajectory segments. Each trajectory segment begins and ends at rest.

      1. A trajectory to move the gripper from its initial configuration to a "standoff" configuration a few cm above the block.
      2. A trajectory to move the gripper down to the grasp position.
      3. Closing of the gripper.
      4. A trajectory to move the gripper back up to the "standoff" configuration.
      5. A trajectory to move the gripper to a "standoff" configuration above the final configuration.
      6. A trajectory to move the gripper to the final configuration of the object.
      7. Opening of the gripper.
      8. A trajectory to move the gripper back to the "standoff" configuration.

    Args:
        ee_initial_config (4x4 Transformation s->e): The end-effector's initial configuration
        cube_initial_config (4x4 Transformation s->c): The initial configuration of the cube
        cube_final_config (4x4 Transformation s->c): The drop off location for the cube (Place configuration)
        ee_grasping_config (4x4 Transformation c->e): The configuration of the end-effector to grasp the cube
        standoff_config (4x4 Transformation c->e): The configuration of the end-effector to prepare to pick/place the cube
        num_reference_configs (int): The number of trajectory reference configs per 0.01 seconds
        debug (bool): If True, print debug information

    Returns:
      tuple of trajectory
    """
    total_time = num_reference_configs * 0.01  # [s]
    # Each trajectory has a time segment alloted to it and
    # the sum of the segments should equal the total time
    traj_1_time = 0.2 * total_time  # Initial -> Pick_Standoff
    traj_2_time = 0.05 * total_time  # Pick_Standoff -> Cube_Initial
    traj_3_time = 0.05 * total_time  # Cube_Initial -> Gripper Closed
    traj_4_time = 0.2 * total_time  # Gripper Closed -> Pick_Standoff
    traj_5_time = 0.2 * total_time  # Pick_Standoff -> Place_Standoff
    traj_6_time = 0.05 * total_time  # Place_Standoff -> Cube_Final
    traj_7_time = 0.05 * total_time  # Cube_Final -> Gripper Open
    traj_8_time = 0.2 * total_time  # Gripper Open -> Place_Standoff
    trajectory_time = np.array([
        traj_1_time, traj_2_time, traj_3_time, traj_4_time,
        traj_5_time, traj_6_time, traj_7_time, traj_8_time
    ])
    # traj_time = total_time / 8
    assert trajectory_time.sum() == total_time, f'Trajectory time does not sum to total time: {
        trajectory_time.sum()} != {total_time}'
    # steps per trajectory segment
    # traj_steps = np.array([
    #     int((t / total_time) * num_reference_configs) for t in trajectory_time
    # ])
    # assert traj_steps.sum() == num_reference_configs, f'Trajectory steps does not sum to total steps: {
    #     traj_steps.sum()} != {num_reference_configs}'
    traj_steps = num_reference_configs // 8
    print(
        f'Generating Trajectory with {num_reference_configs} reference configurations ' +
        f'with {traj_steps} steps per segment and ' +
        f'{total_time} total seconds:\n' +
        '\n'.join([f'\tSegment {i+1}: {t} [s]' for i,
                   t in enumerate(trajectory_time)])
    )
    # Apply transformations for the waypoints we'll need
    pick_standoff_config = cube_initial_config @ standoff_config
    place_standoff_config = cube_final_config @ standoff_config
    pick_grasp_config = cube_initial_config @ ee_grasping_config
    place_grasp_config = cube_final_config @ ee_grasping_config
    # Gripper states for the trajectory
    gripper_states = []
    # Trajectory 1: Initial -> Pick_Standoff (Screw Trajectory)
    gripper_states.extend([0] * traj_steps)
    traj_1 = mr.ScrewTrajectory(
        ee_initial_config, pick_standoff_config, traj_1_time, traj_steps, method=3
    )
    print(
        f'Trajectory 1: Initial -> Standoff\n{np.round(traj_1, 2)}'
    ) if debug else None
    # Trajectory 2: Pick_Standoff -> Cube_Initial (Cartesian Trajectory)
    gripper_states.extend([0] * traj_steps)
    traj_2 = mr.CartesianTrajectory(
        pick_standoff_config, pick_grasp_config, traj_2_time, traj_steps, method=3
    )
    print(
        f'Trajectory 2: Standoff -> Grasp\n{traj_2}'
    ) if debug else None
    # Trajectory 3: Cube_Initial -> Gripper Closed
    gripper_states.extend([1] * traj_steps)
    traj_3 = mr.ScrewTrajectory(
        pick_grasp_config, pick_grasp_config, traj_3_time, traj_steps, method=3
    )
    print(
        f'Trajectory 3: Grasp -> Gripper Closed\n{np.round(traj_3, 2)}'
    ) if debug else None
    # Trajectory 4: Gripper Closed -> Pick_Standoff (Cartesian Trajectory)
    gripper_states.extend([1] * traj_steps)
    traj_4 = mr.CartesianTrajectory(
        pick_grasp_config, pick_standoff_config, traj_4_time, traj_steps, method=3
    )
    print(
        f'Trajectory 4: Gripper Closed -> Standoff\n{np.round(traj_4, 2)}'
    ) if debug else None
    # Trajectory 5: Pick_Standoff -> Place_Standoff (Screw Trajectory)
    gripper_states.extend([1] * traj_steps)
    traj_5 = mr.ScrewTrajectory(
        pick_standoff_config, place_standoff_config, traj_5_time, traj_steps, method=3
    )
    print(
        f'Trajectory 5: Standoff -> Place Standoff\n{np.round(traj_5, 2)}'
    ) if debug else None
    # Trajectory 6: Place_Standoff -> Cube_Final (Cartesian Trajectory)
    gripper_states.extend([1] * traj_steps)
    traj_6 = mr.CartesianTrajectory(
        place_standoff_config, place_grasp_config, traj_6_time, traj_steps, method=3
    )
    print(
        f'Trajectory 6: Place Standoff -> Final\n{np.round(traj_6, 2)}') if debug else None
    # Trajectory 7: Cube_Final -> Gripper Open
    gripper_states.extend([0] * traj_steps)
    traj_7 = mr.ScrewTrajectory(
        place_grasp_config, place_grasp_config, traj_7_time, traj_steps, method=3
    )
    print(
        f'Trajectory 7: Final -> Gripper Open\n{traj_7}'
    ) if debug else None
    # Trajectory 8: Gripper Open -> Place_Standoff
    gripper_states.extend([0] * traj_steps)
    traj_8 = mr.CartesianTrajectory(
        place_grasp_config, place_standoff_config, traj_8_time, traj_steps, method=3
    )
    print(
        f'Trajectory 8: Gripper Open -> Place Standoff\n{np.round(traj_8, 2)}'
    ) if debug else None
    trajectory = np.concatenate(
        (traj_1, traj_2, traj_3, traj_4, traj_5, traj_6, traj_7, traj_8), axis=0
    )
    print(f'Successfully generated trajectory with shape {trajectory.shape}')
    return trajectory, gripper_states


# Initial EE Configuration (T_se, initial)
ee_initial_config = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.5],
    [0, 0, 0, 1]
])

# Initial Cube Configuration (T_sc, initial)
cube_initial_config = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0.025],
    [0, 0, 0, 1]
])

# Final Cube Configuration (T_sc, final)
cube_final_config = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, -1],
    [0, 0, 1, 0.025],
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


trajectory, gripper_states = trajectory_generator(
    ee_initial_config,
    cube_initial_config,
    cube_final_config,
    ee_grasping_config,
    standoff_config,
    num_reference_configs=1024,
    debug=False
)

# Convert trajectory to simulation state
sim_state = traj_to_sim_state(
    trajectory,
    gripper_states,
    write_to_file=True,
    filename='trajectory.csv'
)

assert sim_state.shape == (1024, 13), f'Invalid simulation state shape {
    sim_state.shape}'
