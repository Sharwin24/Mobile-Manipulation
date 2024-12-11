# ME449-Final-Project

## Running the Program
To run the program and generate the trajectories, plots, and log files, run `python3 code/main.py` (if you're in the top level directory). The main function will initialize the robot at a configuration with some error as well as the cube's position (depending on the task). The reference trajectory will be generated and the controller along with the state function will be used to control the robot accordingly.

In `main.py`, the parameters for each simulation can be tuned by adjusting the values within the dictionary `sim_control_params`: 

```python
# control_type options: ['FF+PI', 'P', 'PI']
sim_control_params = {
    "best": {'Kp': np.eye(6) * 2.0, 'Ki': np.eye(6) * 0.01, 'control_type': 'FF+PI'},
    "overshoot": {'Kp': np.eye(6) * 5.0, 'Ki': np.zeros((6, 6)), 'control_type': 'P'},
    "newTask": {'Kp': np.eye(6) * 5.0, 'Ki': np.eye(6) * 0.1, 'control_type': 'FF+PI'}
}
```

If you want to run a bash script to run the python file for you, run `runscript.sh`

### Generated Artifacts
In each simulation (individual folder in the `results/` directory), the following is included:

#### - README_*sim_name*.md
Includes information about the type of controller, the feedback gains, and other useful info about the results of this simulation.

#### - *sim_name*_log.txt
A log file that captures all `stdout` output from running the simulation. Expect to see logs from generating the trajectories and files required for CoppeliaSim to run.

#### - *sim_name_sim.mp4
A video of the simulation in CoppeliaSim

#### - error_plot.png
A figure showing each of the error components of the 6-D error twist `X_err`. The figure also includes a second plot with just a subsection of the timesteps shown to easier see the error response in the beginning.

#### - errors.csv
The list of `X_err` values used to plot the data, saved as a csv.

#### - robot_states.csv
The csv file containing the list of robot states CoppeliaSim needs to run Scene6.

#### - robot_states.png
A figure showcasing the chassis trajectory and end-effector trajectory, as well as the arm joint angles and wheel angles over time.

### - sim_trajectory.csv
The reference trajectory generated from `trajectory_generator` in a csv file. This isn't used since the reference trajectory is directly passed to the main loop in `main` in order to implement the controller. However, it lives here for debugging purposes.

## Robot Constants
The `robot_constants.py` file details all the robot-related variables and offers transformation functions between the space frame and the robot's base/body/end-effector frames. The class is externalized with a global instance so only one object is referenced at a time in case the state is manipulated.

## Trajectory Generator
In order to generate the trajectory for the pick/place task, the `trajectory_generator.py` file takes in a series of transformations detailing the motion's waypoints, builds a trajectory satisfying the waypoints, and writes the end-effector's configuration data over the trajectory to a file. Run the file with no arguments:

```bash
# If you are in the top level directory
python3 code/trajectory_generator.py
```

Then use the CSV file generated in the `reults/` directory (`trajectory.csv`) in order to play the trajectory in CoppeliaSim in Scene8.

## Feedback Controller
The `feedback_controller.py` file contains the task-space feedforward with feedback controller. The function `feedback_control` accepts the control type as a parameter in case the simulation wants to use either a feedforward with feedback, P, or PI controller. This file also provides a method to compute the robot's speeds using the mobile manipulator jacobian (from the RobotConstants class) and a given Twist.

```python
# In robot_constants.py
def Je(self, arm_thetas: np.array) -> np.array:
    # 6x5 Jacobian Matrix for the arm
    J_arm = mr.JacobianBody(self.B, arm_thetas)
    # 6x4 Jacobian Matrix for the base
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
```

## Next State
The file `next_state.py` offers the function `next_state()` which accepts a current robot state as a 13-vector (chassis config, arm config, wheel config) and performs Euler Integration and Odometry to interpolate the next robot state after the given timestep. This function also clips the speeds of the robot based on the maximum speeds of the joints (Saved in RobotConstants)