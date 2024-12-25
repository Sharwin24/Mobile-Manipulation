import numpy as np
import modern_robotics as mr
from robot_constants import RC


def feedback_control(X, Xd, Xd_next, Kp, Ki, control_type: str = 'FF+PI', dt: float = 0.01) -> tuple:
    """
    Calculate the kinematic task-space feedforward plus feedback control law.

    V(t) = [Adjoint(inv(X)*Xd)]V_d(t) + Kp*X_err(t) + Ki*integral(X_err(t))

    Args:
        X (np.array): The current actual end-effector configuration (T_se)
        Xd (np.array): The current end-effector reference configuration (T_se,d)
        Xd_next (np.array): The end-effector reference configuration at the next timestep in the reference trajectory
        Kp (np.array): The Proportional gain matrix
        Ki (np.array): The Integral gain matrix
        control_type (str, optional): The type of simulation. Defaults to 'FF+PI'. ['FF+PI', 'P', 'PI']
        dt (float, optional): The timestep between reference trajectory configs. Defaults to 0.01.

    Returns:
        tuple: A tuple containing the twist V and the error X_err
    """
    X_err = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X) @ Xd))
    # print(f'X_err:\n{np.round(X_err, 3)}')
    Vd = mr.se3ToVec((1/dt) * mr.MatrixLog6(mr.TransInv(Xd) @ Xd_next))
    # print(f'Vd:\n{np.round(Vd, 3)}')
    if control_type == 'FF+PI':
        V = mr.Adjoint(mr.TransInv(X) @ Xd) @ Vd + \
            (Kp @ X_err) + (Ki @ (X_err * dt))
    elif control_type == 'P':
        V = (Kp @ X_err)
    elif control_type == 'PI':
        V = (Kp @ X_err) + (Ki @ (X_err * dt))
    else:
        print(f'Invalid sim_type: {control_type}, using FF+PI')
        V = mr.Adjoint(mr.TransInv(X) @ Xd) @ Vd + \
            (Kp @ X_err) + (Ki @ (X_err * dt))
    # print(f'V:\n{np.round(V, 3)}')
    return V, X_err


def compute_robot_speeds(V: np.array, arm_thetas: np.array, violated_joints: list[int] = []) -> np.array:
    """
    Compute the robot speeds given the twist and arm joint angles.

    Args:
        V (np.array): The twist applied to the robot
        arm_thetas (np.array): The arm joint angles [rad]
        violated_joints (list[int], optional): The list of joints (by joint number) that break joint limits. Defaults to [].

    Returns:
        np.array: The robot speeds as [wheel_speeds, arm_speeds]
    """
    return np.linalg.pinv(RC.Je(arm_thetas, violated_joints)) @ V


def test_joint_limits(arm_thetas: np.array) -> list[bool]:
    """
    Test if the given joint angles are within their joint limits (for each joint).

    Each bool is True if that joint is within its limit

    Args:
        arm_thetas (np.array): The list of 5 arm joint angles [rad]

    Returns:
        list[bool]: A list of 5 bools indicating if each joint is within its limits
    """
    return [j_min < j < j_max for j, (j_min, j_max) in zip(arm_thetas, RC.joint_limits)]


def main():
    # Robot Config: [phi, x, y, theta1, theta2, theta3, theta4, theta5]
    robot_config = np.array([0, 0, 0, 0, 0, 0.2, -1.6, 0])
    X_from_config = RC.T_se(
        x=robot_config[1], y=robot_config[2], phi=robot_config[0],
        arm_thetas=robot_config[3:]
    )
    twist, error = feedback_control(
        X=X_from_config,
        Xd=np.array([
            [0, 0, 1, 0.5],
            [0, 1, 0, 0],
            [-1, 0, 0, 0.5],
            [0, 0, 0, 1]
        ]),
        Xd_next=np.array([
            [0, 0, 1, 0.6],
            [0, 1, 0, 0],
            [-1, 0, 0, 0.3],
            [0, 0, 0, 1]
        ]),
        Kp=np.zeros((6, 6)),
        Ki=np.zeros((6, 6)),
        dt=0.01
    )
    robot_speeds = compute_robot_speeds(
        V=twist,
        arm_thetas=robot_config[3:]
    )
    print(f'Robot Speeds:\n{np.round(robot_speeds, 3)}')


if __name__ == '__main__':
    main()
