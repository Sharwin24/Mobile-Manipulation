import numpy as np
import modern_robotics as mr
from robot_constants import RC


def feedback_control(X, Xd, Xd_next, Kp, Ki, dt):
    # Calculate the kinematic task-space feedforward plus feedback control law,
    # written as:
    # V(t) = [Adjoint(inv(X)*Xd)]V_d(t) + Kp*X_err(t) + Ki*integral(X_err(t))
    # X = The current actual end-effector configuration (T_se)
    # Xd = The current end-effector reference configuration (T_se,d)
    # Xd_next = The end-effector reference configuration at the next timestep in the reference trajectory at a time dt later
    # Kp = The P gain matrix, Ki = The I gain matrix
    # dt = The timestep between reference trajectory configurations
    X_err = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X) @ Xd))
    print(f'X_err:\n{np.round(X_err, 3)}')
    Vd = mr.se3ToVec((1/dt) * mr.MatrixLog6(mr.TransInv(Xd) @ Xd_next))
    print(f'Vd:\n{np.round(Vd, 3)}')
    V = mr.Adjoint(mr.TransInv(X) @ Xd) @ Vd + \
        (Kp @ X_err) + (Ki @ (X_err * dt))
    print(f'V:\n{np.round(V, 3)}')
    return V


def compute_joint_speeds(V, arm_thetas):
    print(f'Je:\n{np.round(RC.Je(arm_thetas), 3)}')
    return np.linalg.pinv(RC.Je(arm_thetas)) @ V


def main():
    # Robot Config: [phi, x, y, theta1, theta2, theta3, theta4, theta5]
    robot_config = np.array([0, 0, 0, 0, 0, 0.2, -1.6, 0])
    X_from_config = RC.T_se(
        x=robot_config[1], y=robot_config[2], phi=robot_config[0],
        arm_thetas=robot_config[3:]
    )
    twist = feedback_control(
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
    robot_speeds = compute_joint_speeds(
        V=twist,
        arm_thetas=robot_config[3:]
    )
    print(f'Robot Speeds:\n{np.round(robot_speeds, 3)}')


if __name__ == '__main__':
    main()
