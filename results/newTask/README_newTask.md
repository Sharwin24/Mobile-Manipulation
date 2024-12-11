# NewTask Simulation
This simulation started the robot at the configuration:

```python
# [phi, x, y, theta1, theta2, theta3, theta4, theta5, wheel1, wheel2, wheel3, wheel4]
[0.523, 0.1, 0.2, 0, -0.587, -0.9, 0, 0, 0, 0, 0, 0]
```

The block's configurations were adjusted from the other simulations:

```python
initial_pose = [1, 0.2, 0.4] # [x, y, theta]
final_pose = [0, -1, -2] # [x, y, theta]
```

The controller used was a **Feedforward with PI Feedback** with the gains:

$$K_P=\begin{bmatrix}5 & 0 & 0 \\ 0 & 5 & 0 \\ 0 & 0 & 5\end{bmatrix}$$
$$K_I=\begin{bmatrix}0.1 & 0 & 0 \\ 0 & 0.1 & 0 \\ 0 & 0 & 0.1\end{bmatrix}$$

This simulation had slightly erratic behavior but was able to correctly grasp the block and reduce error througout the trajectory.