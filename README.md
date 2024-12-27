# Pick and Place Robot System

This project implements a comprehensive pick-and-place system for a robotic arm, designed to handle both static and dynamic blocks in simulated and real-world environments. The system is capable of detecting, grasping, and stacking blocks onto a designated platform under both stationary and motion-based conditions.

## Features

- Handles static and dynamic block manipulation
- Adapts to team-specific configurations (red or blue)
- Utilizes forward and inverse kinematics for precise movements
- Implements real-time detection of AprilTags for pose estimation
- Employs dynamic motion prediction for blocks on a spinning table
- Optimizes performance through joint-angle caching and safe movement planning

## System Overview

System Overview

The system uses a set of criteria for dynamic block picking, as illustrated in the image above. This ensures accurate and reliable grasping of moving blocks on the spinning table.

## Performance

The system demonstrates high accuracy in handling both static and dynamic blocks:

Static Blocks
*Figure 1: 4 Static blocks stacked*

Dynamic Blocks
*Figure 2: 4 Dynamic Blocks*

Combined Stack
*Figure 3: 6 Stacked blocks (combined static and dynamic)*

<img width="656" alt="image" src="https://github.com/user-attachments/assets/719876cf-0f0c-4723-9cf3-35f33b63631f" />


## Evaluation Metrics

| Parameter | Value (Average) |
|-----------|-----------------|
| Number of static blocks picked | 4.0 |
| Number of dynamic blocks picked | 3.5 |
| Total time for algorithm execution (seconds) | 455.5 |
| Average height of static block stack (cm) | 20.0 |
| Average height of dynamic block stack (cm) | 15 |
| Combined average height of both stacks (cm) | 25 |

## System Architecture

The system is built around the `PickAndPlace` class, which encapsulates all functionalities for organized and maintainable code. Key components include:

- Class initialization with team color and hardware interfaces
- World and base frame configuration
- End-effector pose definitions for safe movements
- Block detection and validation methods
- Grasping strategies for static and dynamic blocks
- Tower building and block placement logic

## Challenges and Solutions

- **Angular Velocity Discrepancy**: Addressed by implementing dynamic estimation of the spinning table's speed.
- **Gripper Timing and Force Feedback**: Compensated with alternative sensing and control strategies.
- **Environmental Variability**: Handled through robust image processing and adaptive sensing techniques.

## Logging and Debugging

The system incorporates comprehensive logging for troubleshooting and performance analysis:

Logs
*Figure 4: System logs for performance analysis*
<img width="818" alt="image" src="https://github.com/user-attachments/assets/26941cc6-eecd-4b39-b476-616b269ac98a" />


## Future Improvements

- Implement more advanced path planning algorithms (e.g., RRT)
- Enhance error handling and recovery mechanisms
- Integrate machine learning for improved dynamic object tracking
- Develop alternative feedback mechanisms to compensate for unreliable force feedback

## Conclusion

This pick-and-place system demonstrates robust performance in handling both static and dynamic blocks, showcasing the potential for advanced robotic manipulation in various industrial applications. The project highlights the importance of adaptive strategies and real-time adjustments in robotic systems operating in dynamic environments.

