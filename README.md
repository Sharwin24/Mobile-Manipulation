# ME449-Final-Project

# Robot Constants


# Trajectory Generator
In order to generate the trajectory for the pick/place task, the `trajectory_generator.py` file takes in a series of transformations detailing the motion's waypoints, builds a trajectory satisfying the waypoints, and writes the end-effector's configuration data over the trajectory to a file. Run the file with no arguments:

```bash
# If you are in the top level directory
python3 code/trajectory_generator.py
```

Then use the CSV file generated in the `reults/` directory (`trajectory.csv`) in order to play the trajectory in CoppeliaSim in Scene8.

# Feedback Controller

# Next State