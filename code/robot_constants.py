import numpy as np


class RobotConstants:
    R = 0.0475  # Wheel radius [m]
    L = 0.47 / 2  # Length of Chassis [m]
    W = 0.3 / 2  # Width of the chassis [m]
    H0 = (1 / R) * np.array([
        [-L-W, 1, -1],
        [L+W, 1, 1],
        [L+W, 1, -1],
        [-L-W, 1, 1]
    ])
    # F = pinv(H0)
    F = (R / 4) * np.array([
        [-1 / (L + W), 1/(L + W), 1/(L + W), -1/(L + W)],
        [1, 1, 1, 1],
        [-1, 1, -1, 1]
    ])
    # M_0e = end effector with respect to base frame when the arm is at the home configuration
    M = np.array([
        [1, 0, 0, 0.033],
        [0, 1, 0, 0],
        [0, 0, 1, 0.6546],
        [0, 0, 0, 1]
    ])

    def T_sb(x: float, y: float, phi: float) -> np.array:
        """
        Transformation matrix from the space frame to the base frame.

        Args:
            x (float): x-coordinate of the base frame [m]
            y (float): y-coordinate of the base frame [m]
            phi (float): angle of the base frame [rad]

        Returns:
            np.array: 4x4 Transformation matrix from the space frame to the base frame
        """
        return np.array([
            [np.cos(phi), -np.sin(phi), 0, x],
            [np.sin(phi), np.cos(phi), 0, y],
            [0, 0, 1, 0.0963],
            [0, 0, 0, 1]
        ])


# Global Instance
RC = RobotConstants()
