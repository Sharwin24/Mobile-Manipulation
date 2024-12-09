import numpy as np
import modern_robotics as mr


class RobotConstants:
    """
    Constants and Robot-specific functions for a Mobile Manipulator.

    The mobile base is a mecanum wheeled chassis with 4 wheels.

    The arm is a 5-DOF arm with 5 revolute joints and a gripper for the end-effector.
    """

    @property
    def W(self):
        # Width of Chassis [m]
        return 0.3 / 2

    @property
    def L(self):
        # Length of Chassis [m]
        return 0.47 / 2

    @property
    def R(self):
        # Wheel radius [m]
        return 0.0475

    @property
    def H0(self):
        R = self.R
        L = self.L
        W = self.W
        return (1 / R) * np.array([
            [-L-W, 1, -1],
            [L+W, 1, 1],
            [L+W, 1, -1],
            [-L-W, 1, 1]
        ])

    @property
    def F(self):
        # F = pinv(H0)
        R = self.R
        L = self.L
        W = self.W
        return (R / 4) * np.array([
            [-1 / (L + W), 1/(L + W), 1/(L + W), -1/(L + W)],
            [1, 1, 1, 1],
            [-1, 1, -1, 1]
        ])

    @property
    def M(self):
        # M_0e = end effector with respect to base frame when the arm is at the home configuration
        return np.array([
            [1, 0, 0, 0.033],
            [0, 1, 0, 0],
            [0, 0, 1, 0.6546],
            [0, 0, 0, 1]
        ])

    @property
    def T_b0(self):
        # chassis frame [b] to base frame of arm [0]
        return np.array([
            [1, 0, 0, 0.1662],
            [0, 1, 0, 0],
            [0, 0, 1, 0.0026],
            [0, 0, 0, 1]
        ])

    @property
    def B(self):
        B1 = np.array([0, 0, 1, 0, 0.033, 0])
        B2 = np.array([0, -1, 0, -0.5076, 0, 0])
        B3 = np.array([0, -1, 0, -0.3526, 0, 0])
        B4 = np.array([0, -1, 0, -0.2176, 0, 0])
        B5 = np.array([0, 0, 1, 0, 0, 0])
        return np.array([B1, B2, B3, B4, B5]).T

    def Je(self, arm_thetas: np.array) -> np.array:
        # 6x5 Jacobian Matrix for the arm
        J_arm = mr.JacobianBody(self.B, arm_thetas)
        # 6x4 Jacobian Matrix for the base
        # F_6 = [0_m, 0_m, F, 0_m] 6xm matrix
        F_6 = np.array([
            np.zeros(4),
            np.zeros(4),
            self.F[0],
            self.F[1],
            self.F[2],
            np.zeros(4)
        ])
        # J_base = [Adjoint(inv(T_0e) * inv(T_b0))] * F_6
        J_base = mr.Adjoint(
            mr.TransInv(self.T_0e(arm_thetas)) @ mr.TransInv(self.T_b0)
        ) @ F_6
        # Mobile Manipulator Jacobian
        Je = np.hstack((J_base, J_arm))
        return Je

    def T_sb(self, x: float, y: float, phi: float) -> np.array:
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

    def T_0e(self, arm_thetas: np.array) -> np.array:
        return mr.FKinBody(self.M, self.B, arm_thetas)

    def T_se(self, x: float, y: float, phi: float, arm_thetas: np.array) -> np.array:
        return self.T_sb(x, y, phi) @ self.T_b0 @ self.T_0e(arm_thetas)


# Global Instance
RC = RobotConstants()

# print(
#     np.round(RC.Je(
#         arm_thetas=np.array([0, 0, 0.2, -1.6, 0]),
#         wheel_angles=np.array([0, 0, 0, 0])
#     ), 3)
# )
