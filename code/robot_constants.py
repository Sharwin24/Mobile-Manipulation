import numpy as np
import modern_robotics as mr


class RobotConstants:
    """
    Constants and Robot-specific functions for a Mobile Manipulator.

    The mobile base is a mecanum wheeled chassis with 4 wheels.

    The arm is a 5-DOF arm with 5 revolute joints and a gripper for the end-effector.
    """

    @property
    def W(self) -> float:
        """The Width of the chassis [m]"""
        return 0.3 / 2

    @property
    def L(self) -> float:
        """The Length of the chassis [m]"""
        return 0.47 / 2

    @property
    def R(self) -> float:
        """The wheel radius [m]"""
        return 0.0475

    @property
    def H0(self) -> np.array:
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
    def F(self) -> np.array:
        """F = pinv(H0)"""
        R = self.R
        L = self.L
        W = self.W
        return (R / 4) * np.array([
            [-1 / (L + W), 1/(L + W), 1/(L + W), -1/(L + W)],
            [1, 1, 1, 1],
            [-1, 1, -1, 1]
        ])

    @property
    def M(self) -> np.array:
        """The home configuration of the arm, (M_0e)"""
        return np.array([
            [1, 0, 0, 0.033],
            [0, 1, 0, 0],
            [0, 0, 1, 0.6546],
            [0, 0, 0, 1]
        ])

    @property
    def T_b0(self) -> np.array:
        """Transformation matrix from chassis frame [b] -> base frame of arm [0]"""
        return np.array([
            [1, 0, 0, 0.1662],
            [0, 1, 0, 0],
            [0, 0, 1, 0.0026],
            [0, 0, 0, 1]
        ])

    @property
    def B(self):
        """Screw axes of the arm joints in the end-effector frame"""
        B1 = np.array([0, 0, 1, 0, 0.033, 0])
        B2 = np.array([0, -1, 0, -0.5076, 0, 0])
        B3 = np.array([0, -1, 0, -0.3526, 0, 0])
        B4 = np.array([0, -1, 0, -0.2176, 0, 0])
        B5 = np.array([0, 0, 1, 0, 0, 0])
        return np.array([B1, B2, B3, B4, B5]).T

    @property
    def joint_limits(self) -> np.array:
        """The min/max joint limits for the 5 arm joints [rad]"""
        return np.array(
            [-np.pi, np.pi],  # Joint 1
            [-np.pi, np.pi],  # Joint 2
            [-np.pi, np.pi],  # Joint 3
            [-np.pi, np.pi],  # Joint 4
            [-np.pi, np.pi]  # Joint 5
        )

    def robot_config(self, T_se: np.array) -> np.array:
        """
        Given an end-effector transformation (T_se), return the robot configuration.
        The robot configuration is a 8x1 vector: [phi, x, y, theta1, theta2, theta3, theta4, theta5]

        Args:
            T_se (np.array): The transformation matrix from the space frame to the end-effector frame

        Returns:
            (8x1) np.array: The robot configuration: [phi, x, y, theta1, theta2, theta3, theta4, theta5]
        """
        # Given a transformation from space frame to the end-effector
        # return the robot configuration [phi, x, y, theta1, theta2, theta3, theta4, theta5]
        arm_config = mr.IKinBody(
            Blist=self.B, M=self.M, T=T_se,
            thetalist0=np.array([0, 0, 0, 0, 0]),
            eomg=0.01, ev=0.001
        )
        # phi, x, y
        x = T_se[0, 3]
        y = T_se[1, 3]
        phi = np.arctan2(T_se[1, 0], T_se[0, 0])
        base_config = [phi, x, y]
        return np.hstack((base_config, arm_config))

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

    def T_sb(self, phi: float, x: float, y: float) -> np.array:
        """
        Transformation matrix from the space frame to the base frame.

        Args:
            phi (float): angle of the base frame [rad]
            x (float): x-coordinate of the base frame [m]
            y (float): y-coordinate of the base frame [m]

        Returns:
            (4x4) np.array: Transformation matrix from the space frame to the base frame
        """
        return np.array([
            [np.cos(phi), -np.sin(phi), 0, x],
            [np.sin(phi), np.cos(phi), 0, y],
            [0, 0, 1, 0.0963],
            [0, 0, 0, 1]
        ])

    def T_0e(self, arm_thetas: np.array) -> np.array:
        """
        Transformation matrix from the base frame of the arm to the end-effector frame.

        Args:
            arm_thetas (np.array): The arm joint angles [rad]

        Returns:
            (4x4) np.array: The transformation matrix from the base frame of the arm to the end-effector frame
        """
        return mr.FKinBody(self.M, self.B, arm_thetas)

    def T_se(self, phi: float, x: float, y: float, arm_thetas: np.array) -> np.array:
        """
        Transformation matrix from the space frame to the end-effector frame.

        Args:
            phi (float): The angle of the base frame [rad]
            x (float): The x-coordinate of the base frame [m]
            y (float): The y-coordinate of the base frame [m]
            arm_thetas (np.array): The arm joint angles [rad]

        Returns:
            (4x4) np.array: The transformation matrix from the space frame to the end-effector frame
        """
        return self.T_sb(phi, x, y) @ self.T_b0 @ self.T_0e(arm_thetas)


# Global Instance
RC = RobotConstants()
