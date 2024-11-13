import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from lib.calcJacobian import calcJacobianExpanded


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE

        att_f = np.zeros((3, 1))
        norm = np.linalg.norm(target - current)

        att_force_weight = 2.0
        # att_force_weight = 0.5

        # if norm is above a tolerance, use conic potential - hence unit vector
        tolerance = 0.1 * np.sqrt(len(target))
        # tolerance = 0.5
        if norm > tolerance:
            att_f = (1 / norm) * (target - current) #  l1 norm as potential
        else:
            att_f = att_force_weight * (target - current) # l2 norm as potential

        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE

        rep_f = np.zeros((3, 1)) 
        rep_force_weight = 1e-3
        # rep_force_weight = 0.5

        if len(unitvec) == 0:
            return rep_f

        dist, _ = PotentialFieldPlanner.dist_point2box(current.reshape(1, 3), obstacle)
        box_length_x = obstacle[3] - obstacle[0]
        box_length_y = obstacle[4] - obstacle[1]
        box_length_z = obstacle[5] - obstacle[2]
        box_max_length = max(box_length_x, box_length_y, box_length_z)
        d0 = 1.5 * box_max_length
        d0 = 0.5
        if dist < d0:
            rep_f = -1.0 * rep_force_weight * ((1.0/dist) - (1.0/d0)) * (1.0 / dist**2) * unitvec

        ## END STUDENT CODE

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame - changed to 3x10
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame - changed to 3x10

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector - changed to 3x10
        """

        ## STUDENT CODE STARTS HERE

        joint_forces = np.zeros(target.shape)

        for i in range(joint_forces.shape[1]):
            target_position = target[:, i]
            current_position = current[:, i]
            att_f = PotentialFieldPlanner.attractive_force(target_position, current_position).reshape(3,)
            joint_forces[:, i] = att_f
            for obstacle_num in range(len(obstacle)):
                curr_obstacle = obstacle[obstacle_num]
                _, unitvec = PotentialFieldPlanner.dist_point2box(current_position.reshape(1, 3), curr_obstacle)
                rep_f = PotentialFieldPlanner.repulsive_force(curr_obstacle, current_position, unitvec).reshape(3,)
                joint_forces[:, i] += rep_f

        ## END STUDENT CODE
        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector - changed to 3x10
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE

        joint_torques = np.zeros((1, 9))
        
        # Calculate the jacobians for each joint
        Js = calcJacobianExpanded(q) # 10 x 3 x 9

        # Calculate the torques for each joint
        for i in range(10): # 9 as 0th joint won't have any torque
            curr_joint_torques = np.matmul(np.transpose(Js[i]), joint_forces[:, i])
            joint_torques[0] += curr_joint_torques

        ## END STUDENT CODE

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        # Calculate the positions based on these joints
        # target_positions, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        # current_positions, _ = PotentialFieldPlanner.fk.forward_expanded(current)
        # print(target_positions, current_positions)

        # Calculate the distance
        # distance = np.linalg.norm(target_positions - current_positions)
        distance = np.linalg.norm(target - current)

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct, print_torque = False):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """

        ## STUDENT CODE STARTS HERE

        dq = np.zeros((1, 7))

        # Use the map struct to create obstacles array
        obstacles = map_struct.obstacles

        # Compute the forces
        # Calculate current position of joints, end-effector and virtual joints using calculateFKJac
        current_positions, _ = PotentialFieldPlanner.fk.forward_expanded(q)
        if print_torque:
            print("Current Positions: \n", np.round(current_positions, 2))
        target_positions, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        dF = PotentialFieldPlanner.compute_forces(target_positions.transpose(), obstacles, current_positions.transpose())
        if print_torque:
            print("Forces: \n", np.round(dF, 2))
        torque = PotentialFieldPlanner.compute_torques(dF, q)[0]
        # remove the last 2 virtual joints
        torque = torque[:-2]
        if print_torque:
            print("Torque: ", torque)
        unit_norm_torque = torque / np.linalg.norm(torque)
        dq = unit_norm_torque

        ## END STUDENT CODE

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        q_path = np.array([]).reshape(0,7)
        num_steps = 0
        alpha = 0.1 # learning rate

        while num_steps < self.max_steps:

            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            # Compute gradient 
            # TODO: this is how to change your joint angles 
            dq = self.compute_gradient(start, goal, map_struct, num_steps == 0)
            if (num_steps == 0):
                print("Initial gradient: ", dq)

            # Termination Conditions
            # TODO: check termination conditions
            if np.linalg.norm(dq) < self.min_step_size:
                break # exit the while loop if conditions are met!

            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function
            # Detect collision requires line start and end points,
            # and it checks if the line intersects with the box
            line_start_pts, _ = self.fk.forward_expanded(start)
            line_end_pts, _ = self.fk.forward_expanded(start + alpha*dq)

            mayday_flag = False
            for i in range(len(map_struct.obstacles)):
                # print("map_struct.obstacles[i]: ", map_struct.obstacles[i])
                if np.any(detectCollision(line_start_pts, line_end_pts, map_struct.obstacles[i])):
                    print("Collision detected !!!!")
                    mayday_flag = True
                    break

            if mayday_flag:
                break

            # Update joint angles
            prev_start = start
            start = start + alpha * dq

            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            # Measure q_distance between start and start + dq
            # If distance is less than a threshold, perform random walk
            # Random walk is a small change in the joint angles
            # This is to escape local minima
            # Use np.random.uniform to generate random joint angles
            # while self.q_distance(prev_start, start) < 1e-4:
            #     # Compute random perturations, where lower limit for a
            #     # particular joint is its distance from the lower limit
            #     # and upper limit is the distance from the upper limit
            #     dq = np.random.uniform(self.lower - start, self.upper - start)
            #     dq = dq / np.linalg.norm(dq)
            #     start = start + alpha * dq

            num_steps += 1
            q_path = np.vstack([q_path, start])
            ## END STUDENT CODE

        return q_path

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
