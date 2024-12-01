import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy

from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap

class Tree:

    class Node:
        def __init__(self, state, parent):
            self.state = state
            self.parent = parent


    def __init__(self):
        self.nodes = []

    def __len__(self):
        return len(self.nodes)

    def add_node(self, state, parent):
        node = Tree.Node(state, parent)
        self.nodes.append(node)
        return node

    def nearest_neighbor(self, state, comparator):
        return min(self.nodes, key=lambda node: comparator(node.state, state))

    def extract_path_to_root(self, node):
        path = []
        while node.parent is not None:
            path.append(node.state)
            node = node.parent
        path.append(node.state)
        path.reverse()
        return path


def is_link_collisions(q, map_struct, fk_instance):
    """
    Checks if any of the links (joints[i], joints[i+1]) are in collision with the map.
    """
    joint_pos, _ = fk_instance.forward_expanded(q)
    for i in range(len(map_struct.obstacles)):
        if np.any(detectCollision(joint_pos[:-1], joint_pos[1:], map_struct.obstacles[i])):
            return True

    return False

def is_self_collision(q, fk_instance):
    """
    Checks if the robot is in self collision.
    """
    joint_pos, _ = fk_instance.forward_expanded(q)
    tolerance = 0.01
    for i in range(7):
        for j in range(i + 1, 7):
            if np.linalg.norm(joint_pos[i] - joint_pos[j]) < tolerance:
                return True  

    return False

def check_if_feasible(current, new, map_struct, fk_instance):
    """
    Checks if the new joint configuration is feasible. 

    INPUTS:
    current - 1x7 numpy array containing the current joint angles
    new - 1x7 numpy array containing the desired joint angles
    map_struct - a map struct containing the obstacle box min and max positions
    fk_instance - a forward kinematics instance

    OUTPUTS:
    feasible - boolean. True if the new joint configuration is feasible. False otherwise
    """
    # Check self collision
    if is_self_collision(new, fk_instance):
        return False

    feasible = True

    num_divisions = 10
    for i in range(num_divisions):
        q = current + (new - current) * i / num_divisions
        if is_link_collisions(q, map_struct, fk_instance):
            feasible = False
            break

        if is_self_collision(q, fk_instance):
            feasible = False
            break

    return feasible


ADVANCED = 1
REACHED = 2
def steer(from_config, to_config, step_size):
    """
    Steer from one configuration toward another, respecting step size.
    Returns new configuration.
    """
    diff = to_config - from_config
    distance = np.linalg.norm(diff)
    
    if distance < step_size:
        return to_config, REACHED
    else:
        return from_config + (diff / distance) * step_size, ADVANCED


TRAPPED = 0
def extend(tree, rand_config, step_size, map_struct, fk_instance):
    """
    Extend the tree to the new node rand_node if possible.
    """
    # Find the nearest neighbor in the tree
    nearest_to_rand = tree.nearest_neighbor(rand_config, lambda x, y: np.linalg.norm(x - y))
    
    status = TRAPPED

    # Check if we can extend to the new configuration
    q_new, status = steer(nearest_to_rand.state, rand_config, step_size)
    if check_if_feasible(nearest_to_rand.state, q_new, map_struct, fk_instance):
        q_new_node = tree.add_node(q_new, nearest_to_rand)
        return q_new_node, status
    
    return None, TRAPPED
    

def connect(tree, new_config, step_size, map_struct, fk_instance):
    """
    Connect the tree to the new node new_config if possible.
    """
    # Check if we can extend to the new configuration
    q_new, status = extend(tree, new_config, step_size, map_struct, fk_instance)
    while status == ADVANCED:
        q_new, status = extend(tree, new_config, step_size, map_struct, fk_instance)

    return q_new, status


def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = []
    path.append(start)

    # initialize start tree
    tree_start = Tree()
    tree_start.add_node(start, None)

    # initialize goal tree
    goal_tree = Tree()
    goal_tree.add_node(goal, None)

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    max_iterations = 1000
    step_size = 0.4
    goal_bias = 0.1

    fk = FK_Jac()
    
    num_steps = 0
    reached = False

    # Implementing RRT-connect
    while num_steps < max_iterations:
        # sample a random configuration
        sample = np.random.uniform(low = lowerLim, high = upperLim)
        while not check_if_feasible(start, sample, map, fk):
            sample = np.random.uniform(low = lowerLim, high = upperLim)

        # Choose which tree to extend
        if random.random() < 0.5:
            current_tree = tree_start
            target_tree = goal_tree
        else:
            current_tree = goal_tree
            target_tree = tree_start

        # Apply goal bias / or start bias if current tree is goal tree
        if current_tree is tree_start:
            if random.random() < goal_bias:
                sample = goal
        else:
            if random.random() < goal_bias:
                sample = start

        # See if we can extend the current tree
        q_new_node, status = extend(current_tree, sample, step_size, map, fk)
        if status != TRAPPED:
            # Try to connect to the other tree
            connection_node, status = connect(target_tree, q_new_node.state, step_size, map, fk)
            
            # If we can extend to the new configuration, we have found a path
            if status == REACHED:
                reached = True
                # Extract the path
                path = current_tree.extract_path_to_root(q_new_node)
                path = path + list(reversed(target_tree.extract_path_to_root(connection_node)))
                if current_tree is goal_tree:
                    path = list(reversed(path))
                break

        # update path
        num_steps += 1

    if not reached:
        print("No path found")
        nearest_to_goal = tree_start.nearest_neighbor(goal, lambda x, y: np.linalg.norm(x - y))
        path = tree_start.extract_path_to_root(nearest_to_goal)
    return np.array(path)

if __name__ == '__main__':
    map_struct = loadmap("maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

    map_struct = loadmap("maps/map2.txt")
    # start = np.array([0,-1,0,-2,0,1.57,0])
    start = np.array([0, 0.4, 0, -2.5, 0, 2.7, 0.707])

    # goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    goal = np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    # print(path)