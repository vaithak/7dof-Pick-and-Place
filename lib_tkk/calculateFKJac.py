import numpy as np
from math import pi

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout

        # Positions of the 7 joints (+ end effector) in their respective frames
        self.homJointPositions = np.array([
            [0, 0, 0,      1],
            [0, 0, 0,      1],
            [0, 0, 0.195,  1],
            [0, 0, 0,      1],
            [0, 0, 0.125,  1],
            [0, 0, -0.015, 1],
            [0, 0, 0.051,  1],
            
            # Also adding 2 virtual joints on the gripper, these are in reference to the end effector
            [0, -0.100, -0.105,  1],
            [0, 0.100, -0.105,  1],

            # End-effector
            [0, 0, 0,      1],
        ])
        
        # Matrix of DH parameters for all 7 links of franka arm
        self.params = np.array([
            [0,      -np.pi/2, 0.192, 0],        # + q[0]
            [0,       np.pi/2, 0,     0],        # + q[1]
            [0.0825,  np.pi/2, 0.316, 0],        # + q[2]
            [0.0825,  np.pi/2, 0,     np.pi],    # + q[3]
            [0,      -np.pi/2, 0.384, 0],        # + q[4]
            [0.088,   np.pi/2, 0,     np.pi],    # + q[5]
            [0,       0,       0.21,  -np.pi/4], # + q[6]
        ])

    
    # feel free to define additional helper methods to modularize your solution for lab 1
    def compute_DH_matrix(self, a_i, alpha_i, d_i, theta_i):
        return np.array([
            [np.cos(theta_i), -np.sin(theta_i) * np.cos(alpha_i), np.sin(theta_i) * np.sin(alpha_i), a_i * np.cos(theta_i)],
            [np.sin(theta_i), np.cos(theta_i) * np.cos(alpha_i), -np.cos(theta_i) * np.sin(alpha_i), a_i * np.sin(theta_i)],
            [0, np.sin(alpha_i), np.cos(alpha_i), d_i],
            [0, 0, 0, 1]
        ])


    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here

        jointPositions = np.zeros((10,3))
        T0e = np.zeros((10,4,4))

        # The lowest joint ia 0.141m above the base
        T0e[0] = np.array([
            [1, 0, 0, 0.],
            [0, 1, 0, 0],
            [0, 0, 1, 0.141],
            [0, 0, 0, 1]
        ])
        jointPositions[0] = (T0e[0] @ self.homJointPositions[0])[:3]

        # Loop over all 7 links + 2 virtual joints + 1 end effector
        # We have already incorporated the virtual joints in the homJointPositions
        Ais = self.compute_Ai(q)
        for i in range(9):
            T0e[i+1] = np.matmul(T0e[i], Ais[i])
            jointPositions[i+1] = (T0e[i+1] @ self.homJointPositions[i+1])[:3]
        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x10 np array of unit vectors describing the axis of rotation 
                                 for each joint in the world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        axis_of_rotation_list = []

        # First fetch Ais for all links + 2 virtual joints + 1 end effector
        Ais = self.compute_Ai(q)
        
        # For each Ai, the z-axis in their corresponding frame is the first three entries of the third column.
        # Along with this, we also maintain Ri0, the rotation matrix from the ith link frame to the base frame
        # The axis of rotation in the world frame will be R0i @ z_i
        axis_of_rotation_list.append(np.array([0, 0, 1]))
        Ri0 = np.identity(3)
        for i in range(9):
            zi = Ais[i][:3, 2]
            axis_of_rotation_list.append(Ri0 @ zi)
            Ri0 = np.matmul(Ri0, Ais[i][:3, :3])

        axis_of_rotation_list = np.array(axis_of_rotation_list)
        return axis_of_rotation_list.T

    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE

        Ai = []
        for i in range(7):
            Ai.append(self.compute_DH_matrix(self.params[i,0], self.params[i,1], self.params[i,2], self.params[i,3] + q[i]))

        # For virtual joints, the frame is same as the end effector frame
        Ai.append(np.identity(4))
        Ai.append(np.identity(4))

        return Ai
    
if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)
    axis_of_rotation = fk.get_axis_of_rotation(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
    print("Axis of Rotation:\n",axis_of_rotation)