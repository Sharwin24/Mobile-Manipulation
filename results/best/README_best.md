# Best Simulation
This simulation started the robot at the configuration:

```python
# [phi, x, y, theta1, theta2, theta3, theta4, theta5, wheel1, wheel2, wheel3, wheel4]
[0.523, 0.1, 0.2, 0, -0.587, -0.9, 0, 0, 0, 0, 0, 0]
```

The controller used was a **Feedforward with PI Feedback** with the gains:

$$K_P=\begin{bmatrix}2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 2\end{bmatrix}$$
$$K_I=\begin{bmatrix}0.01 & 0 & 0 \\ 0 & 0.01 & 0 \\ 0 & 0 & 0.01\end{bmatrix}$$