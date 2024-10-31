import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    q_in = q_in.squeeze()

    fk = FK()
    joint_positions, T0e = fk.forward(q_in)
    axis_of_rotation = fk.get_axis_of_rotation(q_in)

    # Last three rows are angular velocity, can be computed from axis of rotation
    J[3:6, :] = axis_of_rotation

    # For each joint, linear velocity is cross product of axis of rotation, with (o_n - o_i)
    for i in range(7):
        J[0:3, i] = np.cross(axis_of_rotation[:,i], joint_positions[-1] - joint_positions[i])

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
