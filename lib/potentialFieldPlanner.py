import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from lib.calcJacobian import calcJacobianExpanded
from collections import deque


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()


    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5, hyperparams=None):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """
         # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

        # Default hyperparameters
        self.default_params = {
            # Learning rate
            'alpha': lambda iter_num: 0.02,
            
            # Attractive force parameters
            'att_f_region_switch': lambda _: 0.12,
            'att_f_weight': lambda joint_num: 30 if joint_num < 7 else 10,
            
            # Repulsive force parameters
            'rep_f_weight': lambda _: 0.001,
            'rep_f_threshold': lambda _: 0.12,
            
            # Random sampling parameters
            'goal_sample_prob': lambda _: 0.1,
            'random_walk_tries': lambda _: 10,
            
            # Local minima detection
            'min_gradient_norm': lambda _: 1e-2,
        }
        
        # Update with user-provided parameters
        if hyperparams is not None:
            self.params = {**self.default_params, **hyperparams}
        else:
            self.params = self.default_params


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current, att_f_region_switch, att_f_weight, joint_num):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        att_f_region_switch - float representing the threshold for switching from quadratic
        to conic potential

        att_f_weight - lambda function representing the weight of the attractive force according to
        the joint number

        joint_num - int representing the joint number

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """
        att_f = (target - current)
        
        # Choose between conic potential or quadratic potential
        if ((np.linalg.norm(att_f)**2) < att_f_region_switch):
            # use quadratic potential
            att_f = att_f_weight(joint_num) * att_f
        else:
            # use conic potential
            att_f = att_f / np.linalg.norm(att_f)
        
        return att_f


    @staticmethod
    def repulsive_force(obstacle, current, unitvec, rep_f_weight, rep_f_threshold):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        rep_f_weight - float representing the weight of the repulsive force
        rep_f_threshold - float representing the threshold for switching from quadratic
        to conic potential

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """
        rep_f = np.zeros((3, 1))

        if np.linalg.norm(unitvec) == 0:
            print("unitvec is 0")
            return rep_f

        dist, _ = PotentialFieldPlanner.dist_point2box(current.reshape(1, 3), obstacle)
        if dist < rep_f_threshold:
            rep_f = -1.0 * rep_f_weight * ((1.0/dist) - (1.0/rep_f_threshold)) * (1.0 / dist**2) * unitvec.reshape(3, 1)

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


    def compute_forces(self, target, obstacle, current, iter_num):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame - changed to 3x10
        in the world frame - changed to 3x10
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame - changed to 3x10
        iter_num - int representing the current iteration number

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector - changed to 3x10
        """
        joint_forces = np.zeros(target.shape)
        
        att_f_region_switch = self.params['att_f_region_switch'](iter_num)
        rep_f_weight = self.params['rep_f_weight'](iter_num)
        rep_f_threshold = self.params['rep_f_threshold'](iter_num)

        for i in range(joint_forces.shape[1]):
            target_position = target[:, i]
            current_position = current[:, i]
            
            # Compute attractive forces
            att_f = self.attractive_force(
                target_position, 
                current_position,
                att_f_region_switch,
                self.params['att_f_weight'],
                i
            ).reshape(3,)
            
            joint_forces[:, i] = att_f
            
            # Compute repulsive forces
            for obstacle_num in range(len(obstacle)):
                curr_obstacle = obstacle[obstacle_num]
                _, unitvec = self.dist_point2box(current_position.reshape(1, 3), curr_obstacle)
                rep_f = self.repulsive_force(
                    curr_obstacle,
                    current_position,
                    unitvec.reshape(1, 3),
                    rep_f_weight,
                    rep_f_threshold
                ).reshape(3,)
                joint_forces[:, i] += rep_f

        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q, iter_num):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector - changed to 3x10
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
        for i in range(10):
            curr_joint_torques = np.matmul(np.transpose(Js[i]), joint_forces[:, i])
            joint_torques[0] += curr_joint_torques
            if (i == 0 or i == 1) and iter_num == 0:
                print(joint_torques)

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
        # Calculate the distance
        distance = np.linalg.norm(target - current)

        ## END STUDENT CODE

        return distance

    def compute_gradient(self, q, target, map_struct, iter_num):
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
        target_positions, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        dF = self.compute_forces(target_positions.transpose(), obstacles, current_positions.transpose(), iter_num)
        
        # Compute the torques
        torque = PotentialFieldPlanner.compute_torques(dF, q, iter_num)[0]
        # remove the last 2 virtual joints
        torque = torque[:-2]
        unit_norm_torque = torque # / np.linalg.norm(torque)
        dq = unit_norm_torque

        if iter_num == 0:
            print("Current Positions: \n", np.round(current_positions, 2))
            print("Target Positions: \n", np.round(target_positions, 2))
            print("Forces: \n", np.round(dF, 2))
            print("Torque: \n", np.round(torque, 2))
            print("Gradient: \n", np.round(dq, 2))

        ## END STUDENT CODE

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def is_link_collisions(self, q, map_struct):
        """
        Checks if any of the links (joints[i], joints[i+1]) are in collision with the map.
        """
        joint_pos, _ = self.fk.forward_expanded(q)
        for i in range(len(map_struct.obstacles)):
            if np.any(detectCollision(joint_pos[:-1], joint_pos[1:], map_struct.obstacles[i])):
                return True

        return False

    def is_self_collision(self, q):
        """
        Checks if the robot is in self collision.
        """
        joint_pos, _ = self.fk.forward_expanded(q)
        tolerance = 0.01
        for i in range(7):
            for j in range(i + 1, 7):
                if np.linalg.norm(joint_pos[i] - joint_pos[j]) < tolerance:
                    return True  

        return False

    def check_if_feasible(self, current, new, map_struct):
        """
        Checks if the new joint configuration is feasible. 

        INPUTS:
        current - 1x7 numpy array containing the current joint angles
        new - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        feasible - boolean. True if the new joint configuration is feasible. False otherwise
        """
        # Check self collision
        if self.is_self_collision(new):
            return False

        # Check link collision
        if self.is_link_collisions(new, map_struct):
            return False

        feasible = True

        num_divisions = 10
        for i in range(num_divisions):
            q = current + (new - current) * i / num_divisions
            if self.is_link_collisions(q, map_struct):
                feasible = False
                break

            if self.is_self_collision(q):
                feasible = False
                break

        return feasible


    def feasile_random_direction(self, current, map_struct, alpha, num_tries):
        """
        Returns a feasible random direction. 

        INPUTS:
        current - 1x7 numpy array containing the current joint angles
        map_struct - a map struct containing the obstacle box min and max positions
        num_tries - number of tries to generate a feasible direction

        OUTPUTS:
        new - 1x7 numpy array containing a feasible random direction
        feasible - boolean. True if the new joint configuration is feasible. False otherwise
        """
        import copy
        
        new = copy.deepcopy(current)
        for i in range(num_tries):
            dq_direction = np.random.uniform(self.lower, self.upper) - new
            dq_direction = dq_direction / np.linalg.norm(dq_direction)
            new_try = new + alpha * dq_direction
            if self.check_if_feasible(new, new_try, map_struct):
                return dq_direction, True
        return null, False


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
        q_path = np.vstack((q_path, start))

        while num_steps < self.max_steps:
            # Check if goal is reached
            if self.q_distance(goal, start) < self.tol:
                break

            # Random goal sampling
            if np.random.uniform() < self.params['goal_sample_prob'](num_steps):
                if self.check_if_feasible(start, goal, map_struct):
                    q_path = np.vstack((q_path, goal))
                    break
            
            # Compute gradient
            dq = self.compute_gradient(start, goal, map_struct, num_steps)
            
            # Check for local minima
            if np.linalg.norm(dq) < self.params['min_gradient_norm'](num_steps):
                print("Local minima - Randomizing direction...")
                dq, feasible = self.feasile_random_direction(
                    start, 
                    map_struct, 
                    self.params['alpha'](num_steps), 
                    self.params['random_walk_tries'](num_steps)
                )
                if not feasible:
                    break
            else:
                dq = dq / np.linalg.norm(dq)

            # Update position
            alpha = self.params['alpha'](num_steps)
            if not self.check_if_feasible(start, start + alpha*dq, map_struct):
                print("Collided!")
                break

            start = start + alpha * dq
            num_steps += 1
            q_path = np.vstack([q_path, start])

        return q_path


################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    # create planner
    tol = 0.01
    max_steps = 500
    min_step_size = 1e-3
    custom_params = {
        # Learning rate
        # 'alpha': lambda iter_num: 0.02 * (0.99 ** iter_num),  # Decaying learning rate
        'alpha': lambda iter_num: 0.02,

        # Attractive force parameters
        'att_f_region_switch': lambda _: 0.12,
        'att_f_weight': lambda joint_num: 30 if joint_num < 7 else 10,
        
        # Repulsive force parameters
        'rep_f_weight': lambda _: 0.001,
        'rep_f_threshold': lambda _: 0.12,

         # Random sampling parameters
        'goal_sample_prob': lambda _: 0.1,
        'random_walk_tries': lambda _: 10,
        
        # Local minima detection
        'min_gradient_norm': lambda _: 1e-2,
        'window_size': lambda _: 10
    }
    planner = PotentialFieldPlanner(tol, max_steps, min_step_size, custom_params)
    
    # inputs 
    map_struct = loadmap("maps/map1.txt")
    map_struct = loadmap("maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    current_position, _ = planner.fk.forward_expanded(start)
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    target_positions, _ = planner.fk.forward_expanded(goal)

    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        # print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))
    error = PotentialFieldPlanner.q_distance(q_path[-1, :], goal)
    print("Final error: ", error)
    print("q path[-1]: ", q_path[-1, :])
    
    Js = calcJacobianExpanded(start)
    # print("Jacobian: \n", np.round(Js, 2))
    # print("q path: ", q_path)