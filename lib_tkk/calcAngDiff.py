import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation 
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    omega = np.zeros(3)
    ## STUDENT CODE STARTS HERE

    # R_des_curr = rotation matrix to transform a point in desired frame to current frame
    # => R_des = R_curr @ R_des_curr
    R_des_curr = np.matmul(R_curr.T, R_des)

    # Calculate axis of rotation for R_des_curr
    skew_omega = 0.5 * (R_des_curr - R_des_curr.T)
    omega = np.array([skew_omega[2, 1], skew_omega[0, 2], skew_omega[1, 0]])

    # Convert to world frame
    omega = np.matmul(R_curr, omega)

    return omega
