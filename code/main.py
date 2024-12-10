import modern_robotics as mr
import numpy as np

from robot_constants import RC
from next_state import simulate
from trajectory_generator import trajectory_generator, traj_to_sim_state

# Scene setup
initial_cube_config = np.array([1, 0, 0])  # [x, y, theta]
final_cube_config = np.array([0, -1, -np.pi/2])  # [x, y, theta]
initial_ee_config = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.5],
    [0, 0, 0, 1]
])


def main():
    pass


if __name__ == '__main__':
    main()
