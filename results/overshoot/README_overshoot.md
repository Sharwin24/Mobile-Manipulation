# Overshoot Simulation
This simulation started the robot at the configuration:

```python
# [phi, x, y, theta1, theta2, theta3, theta4, theta5, wheel1, wheel2, wheel3, wheel4]
[0.523, 0.1, 0.2, 0, -0.587, -0.9, 0, 0, 0, 0, 0, 0]
```

The controller used was **P Feedback** with the gains:

$$K_P=\begin{bmatrix}5 & 0 & 0 \\ 0 & 5 & 0 \\ 0 & 0 & 5\end{bmatrix}$$

This controller would overshoot and maintain a steady-state error for a few of the error characteristics within `X_err` but would stabilize relatively quickly throughout the trajectory. This also had no impact on the robot's ability to grab the block.