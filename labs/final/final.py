import sys
import numpy as np
from copy import deepcopy
from math import pi

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from lib.IK_position_null import IK
from lib.calculateFK import FK

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

DEBUG = True

class PickAndPlace:
    def __init__(self,
                 team,
                 arm, 
                 detector,
                 start_position,
        ):
        self.arm = arm
        self.team = team
        self.detector = detector
        self.IK_solver = IK()
        self.FK_solver = FK()
        self.start_position = start_position

        """
        Fixed params we know about the world.
        - The distance from the origin of the world frame to the 
          base of the robot.
          The base of the robot has coordinates [0, +-0.990, 0],
          + for blue team and - for red team.
          
        - Platform altitude is the height of the platform above the ground.

        - Platform dimensions: 0.25 m x 0.25 m

        - The distance from robot base to center of the platform is 0.562 m
          in the x-axis.

        - Distance from the world frame center to the end of the platform where
          the blocks are placed is 1.159 m in the y-axis.
          Again, + for blue team and - for red team.
        """
        self.world_to_base_y = 0.990
        if team == 'red':
            self.world_to_base_y *= -1

        self.platform_altitude = 0.2
        self.platform_size = 0.25
        self.platform_center_x = 0.562

        self.platform_center_y_world = 1.159
        if team == 'red':
            self.platform_center_y_world *= -1

        self.block_size = 0.05

        # Store H_ee_camera
        self.H_ee_camera = self.detector.get_H_ee_camera()
        self.debug_print(f"H_ee_camera:\n {self.H_ee_camera}")

        # Use the above information to compute the transformation matrix from
        # the base of the robot to the world frame.
        self.H_world_base = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, self.world_to_base_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Compute the safe coordinates above the center of the platform
        # where the static blocks are placed. Keep safe altitude of 0.4 m.
        safe_z = self.platform_altitude + 0.25
        safe_y = self.platform_center_y_world - self.world_to_base_y

        safe_position_ee = np.array([
                    self.platform_center_x - self.platform_size/4, # more back to ensure no collision
                    safe_y,
                    safe_z
                ])
        
        # Define a safe pose for the end-effector above the static platform,
        # this will be useful for moving the arm around. This is in the base frame.
        # Also, the z-axis is pointing down for the end-effector, x-axis is same as
        # the robot base frame and y-axis is also opposite to the robot base frame.
        self.safe_static_ee_pose_base = np.array([
            [1, 0, 0, safe_position_ee[0]],
            [0, -1, 0, safe_position_ee[1]],
            [0, 0, -1, safe_position_ee[2]],
            [0, 0, 0, 1]
        ])
        self.debug_print(f"Safe static block pose in base frame:\n {self.safe_static_ee_pose_base}")

        # Define a safe pose for the end-effector above the tower-building area.
        safe_position_ee = np.array([
                safe_position_ee[0] - self.platform_size/4, # more back to ensure no collision
                -safe_position_ee[1],
                safe_position_ee[2]
            ])
        self.safe_tower_ee_pose_base = np.array([
            [1, 0, 0, safe_position_ee[0]],
            [0, -1, 0, safe_position_ee[1]],
            [0, 0, -1, safe_position_ee[2]],
            [0, 0, 0, 1]
        ])
        self.debug_print(f"Safe tower block pose in base frame:\n {self.safe_tower_ee_pose_base}")

        # Mode to define whether we are aiming for the static block 
        # or the moving block.
        self.mode = 'static'
        self.placed_static_blocks = 0
        self.placed_moving_blocks = 0

        # Cached joint angles
        self.cached_joint_angles = {
            'safe_static_ee_pose_base': np.array([-0.178, -0.113, -0.141, -1.885, -0.016, 1.773, 0.472])
        }


    """
    Convert from end-effector frame to robot base frame
    """
    def ee_to_base(self, ee_pose):
        current_joint_positions = self.arm.get_positions()
        _, H_base_ee = self.FK_solver.forward(current_joint_positions)
        return H_base_ee @ ee_pose # H_base_ee means from ee to base


    """
    Convert from camera frame to robot base frame
    """
    def camera_to_base(self, camera_pose):
        return self.ee_to_base(self.H_ee_camera @ camera_pose)


    """
    Convert from base to world frame
    """
    def base_to_world(self, base_pose):
        return self.H_world_base @ base_pose
    

    """
    Conditional print function for debugging.
    """
    def debug_print(self, message):
        if DEBUG: print(message)


    """
    Detect static blocks on the platform. No velocity information is needed.
    """
    def detect_static_blocks(self):
        static_blocks = []
        for (name, pose) in self.detector.get_detections():
            self.debug_print(f"Detected block: {name} at pose:\n {pose}")
            if self.valid_static_block(pose):
                static_blocks.append((name, pose))
                self.debug_print(f"Block {name} is a valid static block!")
        return static_blocks


    """
    Given a detected block pose in camera frame, check if the pose matches
    the expected pose of a static block. This is useful for filtering out
    false positives.
    """
    def valid_static_block(self, detected_block_pose):
        # Error margin
        error_margin = 0.01

        # Convert to base frame
        block_pose_base = self.camera_to_base(detected_block_pose)
        block_pose_world = self.base_to_world(block_pose_base)

        # Print the block pose in the world frame
        self.debug_print(f"Block pose in world frame:\n {block_pose_world}")

        # Extract the x, y, z coordinates from the pose - these are the coordinates
        # of the block center in the base frame.
        x_world, y_world, z_world = block_pose_world[:3, 3]

        # Check if the block is within the platform
        if abs(x_world - self.platform_center_x) > self.platform_size/2 + error_margin:
            return False
        if abs(y_world - self.platform_center_y_world) > self.platform_size/2 + error_margin:
            return False
        # Z-coordinate from the world frame should be within the 
        # platform altitude + block size +- error margin range.
        if z_world < self.platform_altitude + self.block_size/2 - error_margin:
            return False
        if z_world > self.platform_altitude + self.block_size/2 + error_margin:
            return False
        
        return True


    """
    Given a target pose in base frame, plan a path and execute the path to reach that pose.
    """
    def move_to_target(self, target_pose):
        found_in_cache = False
        solution = None

        # Check in cached joint angles
        if np.allclose(target_pose, self.safe_static_ee_pose_base):
            if 'safe_static_ee_pose_base' in self.cached_joint_angles:
                solution = self.cached_joint_angles['safe_static_ee_pose_base']
                found_in_cache = True

        if not found_in_cache:
            current_joint_positions = self.arm.get_positions()
            solution, rollout, success, __ = self.IK_solver.inverse(
                target_pose, current_joint_positions, method='J_pseudo', alpha=0.5)
            if not success:
                print("Failed to find a solution for the target pose.")
                return
            self.debug_print(f"Joint angles found using IK solver: {solution}")

        # Move to the target pose
        self.arm.safe_move_to_position(solution)
        self.debug_print(f"Moved to target pose:\n {target_pose}")


    """
    Given a static block pose, grasp the block.
    You can assume that the current pose of the robot is on the
    safe position above the static platform.
    """
    def grasp_static_block(self, block_name, block_pose):
        # Convert to base frame
        block_pose_base = self.camera_to_base(block_pose)
        self.debug_print(f"Grasping block {block_name} at pose:\n {block_pose_base}")

        # Fix z-orientation to be pointing down
        block_pose_base[:3, 2] = np.array([0, 0, -1])

        # Move to the block
        self.move_to_target(block_pose_base)

        # Close the gripper and apply some force
        self.arm.exec_gripper_cmd(pos = 0.049, force = 40)

        # Verify the block is grasped
        gripper_state = self.arm.get_gripper_state()
        if gripper_state['force'][0] < 30:
            print(f"Failed to grasp block {block_name}.")
            return False
        
        self.debug_print(f"Grasped block {block_name}.")
        return True
            

    """
    Place the block on the tower.
    You can assume that the current pose of the robot is on the
    safe position above the tower-building area.
    """
    def place_block(self, block_name):
        self.debug_print(f"Placing block {block_name} on the tower.")

        # Find the pose where the block should be placed
        # The block should be placed on top of the tower
        # at the center of the platform. The pose of the end-effector
        # while placing the block is such that z-axis is pointing down,
        # x-axis is same as the robot base frame and y-axis is opposite
        # to the robot base frame.
        placed_blocks_count = self.placed_static_blocks + self.placed_moving_blocks
        block_position_ee = np.array([
            self.platform_center_x,
            -self.platform_center_y_world + self.world_to_base_y,
            self.platform_altitude + self.block_size * (placed_blocks_count + 1)
        ])
        block_pose_ee = np.array([
            [1, 0, 0, block_position_ee[0]],
            [0, -1, 0, block_position_ee[1]],
            [0, 0, -1, block_position_ee[2]],
            [0, 0, 0, 1]
        ])

        # Move to the block placement pose
        self.move_to_target(block_pose_ee)

        # Reduce the force and open the gripper - just more than the block size
        self.arm.exec_gripper_cmd(pos = self.block_size + 0.002, force = 10)

        # Verify the block is placed 
        # TODO maybe use the camera
        
        self.debug_print(f"Placed block {block_name} on the tower.")

        # Move back to the safe position above the tower-building area
        self.move_to_target(self.safe_tower_ee_pose_base)


    """
    The main control loop for the pick and place task.
    """
    def pick_and_place(self):

        # Always first move to the start position after opening the gripper
        self.arm.open_gripper()
        self.arm.safe_move_to_position(self.start_position)

        num_iterations = 1

        while num_iterations > 0:
            num_iterations -= 1

            # Check the mode
            if self.mode == 'static' and self.placed_static_blocks < 4:
                # Move to the safe position above the static platform
                self.move_to_target(self.safe_static_ee_pose_base)

                # Detect blocks
                detected_static_blocks = self.detect_static_blocks()

                # Choose one block to pick
                if len(detected_static_blocks) > 0:
                    # Choose any one block
                    block_name, block_pose = detected_static_blocks[0]
                    self.debug_print(f"Block {block_name} is chosen for pick and place.")

                    if self.grasp_static_block(block_name, block_pose):
                        # Move to the safe position above the static platform
                        self.move_to_target(self.safe_static_ee_pose_base)

                        # Move to the safe position above the tower-building area
                        self.move_to_target(self.safe_tower_ee_pose_base)

                        # Place the block
                        self.place_block(block_name)

                        # Update the count of placed static blocks
                        self.placed_static_blocks += 1
                        self.debug_print(f"Placed {self.placed_static_blocks} static blocks.")
                else:
                    print("No valid static blocks detected.")
                    


if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("pick_and_place_node")

    arm = ArmController()
    detector = ObjectDetector()

    start_position = start_position = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    # Initialize the pick and place class
    pick_and_place = PickAndPlace(team, arm, detector, start_position)
    pick_and_place.pick_and_place()

    # END STUDENT CODE
