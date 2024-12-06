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
        # where the static blocks are placed. Keep safe altitude of 0.25 m
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
                self.platform_center_x - self.platform_size/2, # more back to ensure no collision
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

        # Define a safe intermediate pose for the end-effector above the static platform.
        # This will be useful to first move the arm to this pose before moving to the
        # grasping pose - better for convergence of the IK solver.
        safe_position_ee = np.array([
                self.platform_center_x,
                self.safe_static_ee_pose_base[1, 3],
                self.platform_altitude + self.block_size + 0.15
            ])
        self.safe_intermediate_static_ee_pose_base = np.array([
            [1, 0, 0, safe_position_ee[0]],
            [0, -1, 0, safe_position_ee[1]],
            [0, 0, -1, safe_position_ee[2]],
            [0, 0, 0, 1]
        ])
        self.debug_print(f"Safe intermediate static block pose in base frame:\n {self.safe_intermediate_static_ee_pose_base}")

        # Define a safe pose for the end-effector above the dynamic table area.
        self.spin_table_radius = 0.3048 # from the center of the table
        self.spin_table_world_y = self.spin_table_radius
        if team == 'red':
            self.spin_table_world_y *= -1
        self.spin_table_height = 0.2
        self.spin_table_width = 0.02
        safe_position_ee = np.array([
                0,
                self.spin_table_world_y - self.world_to_base_y,
                self.spin_table_height + self.block_size + 0.15
            ])
        if team == 'red':
            self.safe_dynamic_ee_pose_base = np.array([
                [0, 1, 0, safe_position_ee[0]],
                [1, 0, 0, safe_position_ee[1]],
                [0, 0, -1, safe_position_ee[2]],
                [0, 0, 0, 1]
            ])
        else:
            self.safe_dynamic_ee_pose_base = np.array([
                [0, -1, 0, safe_position_ee[0]],
                [-1, 0, 0, safe_position_ee[1]],
                [0, 0, -1, safe_position_ee[2]],
                [0, 0, 0, 1]
            ])
        self.debug_print(f"Safe dynamic block pose in base frame:\n {self.safe_dynamic_ee_pose_base}")

        # Mode to define whether we are running in simulation or real robot
        self.mode = 'simulation'
        self.placed_static_blocks = 0
        self.placed_moving_blocks = 0

        # Cached joint angles
        self.cached_joint_angles = {
            'red': {
                'safe_static_ee_pose_base': np.array([-0.178, -0.113, -0.141, -1.885, -0.016, 1.773, 0.472]),
                'safe_tower_ee_pose_base': np.array([ 0.048, -0.306,  0.279, -2.073,  0.085, 1.777, 1.082]),
                'safe_intermediate_static_ee_pose_base': np.array([-0.13875,  0.08052, -0.15922, -1.78293,  0.01325,  1.86239,  0.48415]),
                'safe_dynamic_ee_pose_base': np.array([1.326, 0.505, 0.383, -1.026, -0.182, 1.501, 0.867])
            },
            'blue': {
                'safe_static_ee_pose_base': np.array([0.107, -0.115, 0.207, -1.885, 0.024, 1.772, 1.093]),
                'safe_tower_ee_pose_base': np.array([-0.230, -0.295, -0.121, -2.073, -0.0360, 1.780, 0.447]),
                'safe_intermediate_static_ee_pose_base': np.array([0.18411,  0.08003,  0.11209, -1.78294, -0.00929,  1.86243,  1.08388]),
                'safe_dynamic_ee_pose_base': np.array([-1.160, 0.491, -0.580, -1.226, 0.262, 1.645, 0.651])
            }
        }

        # Assume omega of the spin table is 0.0523 rad/s - equivalent to 0.5 rpm
        self.omega_spin_table = 0.0523


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
    Convert coordinates from world frame to base frame
    """
    def coordinates_world_to_base(self, world_coords):
        return np.array([
            world_coords[0],
            world_coords[1] - self.world_to_base_y,
            world_coords[2]
        ])


    """
    Conditional print function for debugging.
    """
    def debug_print(self, message):
        if DEBUG: print(message)


    """
    Detect blocks on the platform.
    """
    def detect_blocks(
            self,
            validity_criteria = 'static', # static or dynamic
        ):
        blocks = []
        measured_times = []

        retry = False
        if validity_criteria == 'dynamic':
            # In this case, we may need to keep looking for a fixed amount of time.
            lookup_max_time = 10.0
            retry = True

        current_time = time_in_seconds()
        while True:
            total_count = 0
            for (name, pose) in self.detector.get_detections():
                curr_measurement_time = time_in_seconds()
                total_count += 1
                self.debug_print(f"Detected block: {name} at pose:\n {pose}")
                if validity_criteria == 'static':
                    if self.valid_static_block(pose):
                        blocks.append((name, pose))
                        measured_times.append(curr_measurement_time)
                        self.debug_print(f"Block {name} is a valid static block!")
                    else:
                        self.debug_print(f"Block {name} is not a valid static block.")
                elif validity_criteria == 'dynamic':
                    if self.valid_dynamic_block(pose):
                        detectable, theta = self.dynamic_block_pickable(pose)
                        if not detectable:
                            self.debug_print(f"Block {name} is not pickable according to dynamic block criteria.")
                            continue
                        blocks.append((name, pose, theta))
                        measured_times.append(curr_measurement_time)
                        self.debug_print(f"Block {name} is a valid dynamic block!")
                    else:
                        self.debug_print(f"Block {name} is not a valid dynamic block.")
            
            # Check if we have found any blocks
            if len(blocks) > 0:
                if validity_criteria == 'dynamic':
                    # Sort the blocks based on the angle theta
                    blocks = sorted(blocks, key=lambda x: x[2])
                    # remove the theta from the blocks before returning
                    blocks = [(name, pose) for name, pose, _ in blocks]
                break

            # Check if we need to retry
            if not retry:
                break

            # Check if we need to keep looking for more blocks
            if time_in_seconds() - current_time > lookup_max_time:
                break

            # Sleep for a while before checking again
            rospy.sleep(1.0)

        self.debug_print(f"Total blocks detected: {total_count}")
        self.debug_print(f"Valid blocks detected: {len(blocks)}")
        return blocks, measured_times


    """
    Given a detected block pose in camera frame, check if the pose matches
    the expected pose of a static block. This is useful for filtering out
    false positives.
    """
    def valid_static_block(self, detected_block_pose):
        # Error margin - TODO: Test on the real robot
        error_margin = 0.01

        # Convert to base frame
        block_pose_base = self.camera_to_base(detected_block_pose)
        block_pose_world = self.base_to_world(block_pose_base)

        # Print the block pose in the world frame
        self.debug_print(f"Block pose in world frame:\n {block_pose_world}")

        # Extract the x, y, z coordinates from the pose - these are the coordinates
        # of the block center in the world frame.
        x_world, y_world, z_world = block_pose_world[:3, 3]

        # Check if the block is within the platform
        if abs(x_world - self.platform_center_x) > self.platform_size/2 + error_margin:
            return False
        if abs(y_world - self.platform_center_y_world) > self.platform_size/2 + error_margin:
            return False
        # Z-coordinate from the world frame should be within the 
        # platform altitude + block size +- error margin range.
        if self.mode == 'simulation':
            if z_world < self.platform_altitude + self.block_size/2 - error_margin:
                return False
        else:
            if z_world < self.platform_altitude + self.block_size - error_margin:
                return False
        
        return True
    

    """
    Given a detected block pose in camera frame, check if the pose matches
    the expected pose of a dynamic block. This is useful for filtering out
    false positives.
    """
    def valid_dynamic_block(self, detected_block_pose):
        # Error margin - TODO: Test on the real robot
        error_margin = 0.01

        # Convert to base frame
        block_pose_base = self.camera_to_base(detected_block_pose)
        block_pose_world = self.base_to_world(block_pose_base)

        # Print the block pose in the world frame
        self.debug_print(f"Block pose in world frame:\n {block_pose_world}")

        # Extract the x, y, z coordinates from the pose - these are the coordinates
        # of the block center in the world frame.
        x_world, y_world, z_world = block_pose_world[:3, 3]

        # Check if the block is within the dynamic table area, the table's center is
        # at the origin of the world frame.
        if abs(x_world) > self.spin_table_radius + error_margin:
            return False
        if abs(y_world) > self.spin_table_radius + error_margin:
            return False
        
        # Z-coordinate from the world frame should be within the
        # table height + block size +- error margin range.
        # TODO: Test on the real robot, it should self.block_size/2 or self.block_size.
        if z_world < self.spin_table_height + self.block_size/2 - error_margin:
            return False
        if z_world > self.spin_table_height + self.block_size/2 + error_margin:
            return False

        return True


    """
    Given a target pose in base frame, plan a path and execute the path to reach that pose.
    """
    def move_to_target(self, target_pose, time_to_reach = -1):
        found_in_cache = False
        solution = None

        # Check in cached joint angles
        if np.allclose(target_pose, self.safe_static_ee_pose_base):
            if 'safe_static_ee_pose_base' in self.cached_joint_angles[self.team]:
                solution = self.cached_joint_angles[self.team]['safe_static_ee_pose_base']
                found_in_cache = True
                self.debug_print(f"Found joint angles in cache for safe static pose:\n {solution}")
        elif np.allclose(target_pose, self.safe_tower_ee_pose_base):
            if 'safe_tower_ee_pose_base' in self.cached_joint_angles[self.team]:
                solution = self.cached_joint_angles[self.team]['safe_tower_ee_pose_base']
                found_in_cache = True
                self.debug_print(f"Found joint angles in cache for safe tower pose:\n {solution}")
        elif np.allclose(target_pose, self.safe_intermediate_static_ee_pose_base):
            if 'safe_intermediate_static_ee_pose_base' in self.cached_joint_angles[self.team]:
                solution = self.cached_joint_angles[self.team]['safe_intermediate_static_ee_pose_base']
                found_in_cache = True
                self.debug_print(f"Found joint angles in cache for safe intermediate static pose:\n {solution}")
        elif np.allclose(target_pose, self.safe_dynamic_ee_pose_base):
            if 'safe_dynamic_ee_pose_base' in self.cached_joint_angles[self.team]:
                solution = self.cached_joint_angles[self.team]['safe_dynamic_ee_pose_base']
                found_in_cache = True
                self.debug_print(f"Found joint angles in cache for safe dynamic pose:\n {solution}")

        if not found_in_cache:
            num_trials = 3
            success = False
            for i in range(num_trials):
                current_joint_positions = self.arm.get_positions()
                # Use the IK solver to find a joint angle solution
                solution, rollout, success, __ = self.IK_solver.inverse(
                    target_pose, current_joint_positions, method='J_pseudo', alpha=0.5)
                if success:
                    break
                self.debug_print(f"Failed to find a solution for the target pose. Retrying...")
                self.debug_print(f"Current joint positions: {current_joint_positions}")

            if not success:
                print("Failed to find a solution for the target pose.")
                return
            
            self.debug_print(f"Joint angles found using IK solver: {solution}")

        # Wait for the required time before moving to the target pose
        if time_to_reach != -1:
            # Consider the time to reach the target pose.
            time_to_move_to_pose = 2 # 2 second for now
            expected_time_to_reach = time_to_reach - time_to_move_to_pose
            while time_in_seconds() < expected_time_to_reach:
                pass
        
        # Move to the target pose
        self.arm.safe_move_to_position(solution)
        
        # Verify the arm moved to the target pose
        self.debug_print(f"Expected moving to target pose:\n {target_pose}")
        actual_joint_positions = self.arm.get_positions()
        actual_ee_pose = self.FK_solver.forward(actual_joint_positions)[1]
        self.debug_print(f"Actual joint angles after moving to target pose:\n {self.arm.get_positions()}")
        self.debug_print(f"Actual end-effector pose after moving to target pose:\n {actual_ee_pose}")
        self.debug_print(f"Error in joint angles after moving to target pose:\n {np.linalg.norm(solution - actual_joint_positions)}")
        self.debug_print(f"Error in end-effector pose after moving to target pose:\n {np.linalg.norm(target_pose - actual_ee_pose)}")


    """
    Manually align end-effector x-axis with the provided axis. 
    Assume that they are both perpendicular to the base z-axis. So,
    only add/subtract angle between them to the last joint angle of the arm.
    You are also provided with the calculated angle between the two axes.
    """
    def manual_x_align(self, desired_x_axis, calculated_angle):
        current_joint_positions = self.arm.get_positions()
        curr_x_axis = np.array([1, 0, 0])
        # Decide the direction of the angle
        if np.cross(curr_x_axis, desired_x_axis)[2] > 0:
            calculated_angle *= -1
        current_joint_positions[-1] += calculated_angle
        self.arm.safe_move_to_position(current_joint_positions)
        self.debug_print(f"Manually aligned x-axis with the provided axis.")


    """
    Find desired end effector pose, given the block pose in camera frame to pick it up.
    The crux is that end-effector should always lift it facing down. Whereas, the block's
    orientation is arbitrary (still orthogonal to the camera frame), but x, y and z can be swapped
    in any order. So, we need to find the correct x, y and z for the end-effector.

    One of the axis is aligned with z or -z, this can be found by checking the max value of 
    3rd row of the pose matrix. The rest two axis should be choosen as x and y for the 
    end-effector frame. This will be useful for the grasp pose.

    Choose x as the one with one of the smallest angle with provided desired x-axis (in base frame).
    """
    def find_desired_ee_pose(self, block_pose, desired_x_axis):
        block_pose_base = self.camera_to_base(block_pose)

        desired_end_effector_pose = deepcopy(block_pose_base)
        chosen_z = np.array([0, 0, -1])
        desired_end_effector_pose[:3, 2] = chosen_z
        self.debug_print(f"Block pose in base frame:\n {block_pose_base}")

        # One of the axis is aligned with z or -z, this can be found
        # by checking the max value of 3rd row of the pose matrix.
        # The rest two axis should be choosen as x and y for the end-effector
        # frame. This will be useful for the grasp pose.
        axis = np.argmax(np.abs(block_pose_base[2, :3]))
        self.debug_print(f"Axis with max value: {axis}")
        chosen_x = None
        best_angle = np.pi
        for i in range(3):
            if i != axis:
                curr_x = block_pose_base[:3, i]
                # Set the z-component to 0
                curr_x[2] = 0
                # Normalize the current x-axis
                curr_x /= np.linalg.norm(curr_x)
                # measure angle between curr_x and [1, 0, 0], assume norm is 1
                curr_angle = np.arccos(np.dot(curr_x, desired_x_axis))
                neg_angle = np.pi - curr_angle
                # Choose the one with the smallest angle out of curr_x and -curr_x
                if neg_angle < curr_angle:
                    curr_angle = neg_angle
                    curr_x = -curr_x
                # Choose the best overall angle
                if curr_angle < best_angle:
                    best_angle = curr_angle
                    chosen_x = curr_x
        chosen_y = np.cross(chosen_z, chosen_x)
        desired_end_effector_pose[:3, 0] = chosen_x
        desired_end_effector_pose[:3, 1] = chosen_y
        return desired_end_effector_pose, chosen_x, best_angle


    """
    Given a static block pose, grasp the block.
    You can assume that the current pose of the robot is on the
    safe position above the static platform.
    """
    def grasp_static_block(self, block_name, block_pose):
        desired_end_effector_pose, chosen_x, best_angle = \
                self.find_desired_ee_pose(block_pose, np.array([1, 0, 0]))
        self.debug_print(f"Desired end-effector pose for grasping block {block_name}:\n {desired_end_effector_pose}")

        # Move to the intermediate pose above the block
        self.move_to_target(self.safe_intermediate_static_ee_pose_base)

        # Manually align the x-axis of the end-effector with the x-axis of the block
        self.manual_x_align(chosen_x, best_angle)

        # Move to the block
        self.move_to_target(desired_end_effector_pose)

        # Close the gripper and apply some force
        self.arm.exec_gripper_cmd(pos = 0.040, force = 50)

        # Verify the block is grasped
        gripper_state = self.arm.get_gripper_state()
        # TODO: Add this on the real robot, simulation is not accurate
        # if gripper_state['force'][0] < 20:
            # print(f"Failed to grasp block {block_name}.")
            # return False
        
        self.debug_print(f"Grasped block {block_name}.")
        return True
    

    """
    Given a moving block pose, grasp the block.
    You can assume that the current pose of the robot is on the
    safe position above the dynamic table area.
    """
    def grasp_moving_block(self, block_name, block_pose, detected_time):
        desired_x_axis = np.array([0, -1, 0])
        if self.team == 'red':
            desired_x_axis = -desired_x_axis
        desired_end_effector_pose, chosen_x, best_angle = \
                self.find_desired_ee_pose(block_pose, desired_x_axis)
        self.debug_print(f"Desired end-effector pose for grasping block {block_name}:\n {desired_end_effector_pose}")

        # We will calculate the future desired end-effector pose when the block crosses the y-axis
        # and then move to that pose to grasp the block.
        desired_end_effector_future_pose = deepcopy(desired_end_effector_pose)

        # Calculate the radius of the block's center from the center of the table
        block_pose_base = self.camera_to_base(block_pose)
        block_pose_world = self.base_to_world(block_pose_base)
        x_world, y_world, z_world = block_pose_world[:3, 3]
        r = np.sqrt(x_world**2 + y_world**2)
        future_box_center_world = np.array([0, r, z_world])
        if self.team == 'red':
            future_box_center_world[1] *= -1
        future_box_center_base = self.coordinates_world_to_base(future_box_center_world)
        desired_end_effector_future_pose[:3, 3] = future_box_center_base

        # Calculate what chosen_x would become when the center of block crosses the y-axis
        theta_rotation_around_z = best_angle
        Rot_z = np.array([
            [np.cos(theta_rotation_around_z), -np.sin(theta_rotation_around_z), 0],
            [np.sin(theta_rotation_around_z), np.cos(theta_rotation_around_z), 0],
            [0, 0, 1]
        ])
        chosen_x_future = Rot_z @ chosen_x
        chosen_y_future = np.cross(np.array([0, 0, -1]), chosen_x_future)
        desired_end_effector_future_pose[:3, 0] = chosen_x_future
        desired_end_effector_future_pose[:3, 1] = chosen_y_future
        desired_end_effector_future_pose[:3, 2] = np.array([0, 0, -1])
        self.debug_print(f"Desired end-effector future pose for grasping block {block_name}:\n {desired_end_effector_future_pose}")

        # Time after which the block will cross the y-axis, when theta_rotation_around_z is 0
        time_to_cross = theta_rotation_around_z / self.omega_spin_table

        # Move to the block with desired end-effector future pose at the time_to_cross + detected_time
        self.move_to_target(desired_end_effector_future_pose, time_to_cross + detected_time)

        # Close the gripper and apply some force
        self.arm.exec_gripper_cmd(pos = 0.040, force = 50)

        # Verify the block is grasped
        gripper_state = self.arm.get_gripper_state()
        # TODO: Add this on the real robot, simulation is not accurate
        # if gripper_state['force'][0] < 20:
            # print(f"Failed to grasp block {block_name}.")
            # return False
        
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

        # Then open the gripper
        self.arm.open_gripper()

        # Verify the block is placed 
        # TODO maybe use the camera - measure the z-coordinate of the topmost block
        # Will be tough, what if the camera detects side view of some bottom april tag.
        
        self.debug_print(f"Placed block {block_name} on the tower.")

        # Move back to the safe position above the tower-building area
        self.move_to_target(self.safe_tower_ee_pose_base)
        return True


    """
    Control loop for static pick and place task.
    Assume that you are starting from the start position and the gripper is open.
    """
    def static_pick_and_place(self):
        # Move to the safe position above the static platform
        self.move_to_target(self.safe_static_ee_pose_base)

        # Detect blocks
        detected_static_blocks, _ = self.detect_blocks(validity_criteria='static')

        # Choose one block to pick
        if len(detected_static_blocks) > 0:
            # Choose any one block
            block_name, block_pose = detected_static_blocks[0]
            self.debug_print(f"Block {block_name} is chosen for static pick and place.")

            if self.grasp_static_block(block_name, block_pose):
                # Move to the safe position above the static platform
                self.move_to_target(self.safe_static_ee_pose_base)

                # Move to the safe position above the tower-building area
                self.move_to_target(self.safe_tower_ee_pose_base)

                # Place the block
                if self.place_block(block_name):
                    self.placed_static_blocks += 1
                    self.debug_print(f"Placed {self.placed_static_blocks} static blocks.")
        else:
            print("No valid static blocks detected.")


    """
    Check if the block pose (in camera frame) is pickable according to the
    dynamic block choosing criteria.
    """
    def dynamic_block_pickable(self, block_pose):
        # TODO: check if this is enough, also test on the real robot
        time_margin = 2.0 + 3.0 # 2 seconds for IK_solver, 3 seconds for moving to the block
        theta_margin = self.omega_spin_table * time_margin

        block_pose_base = self.camera_to_base(block_pose)
        block_pose_world = self.base_to_world(block_pose_base)
        
        # Reject blocks moving away from the y-axis
        x_world, y_world, _ = block_pose_world[:3, 3]
        if self.team == 'red' and x_world > 0:
            return False, None
        elif self.team == 'blue' and x_world < 0:
            return False, None

        # Compute the angle of two vectors:
        # 1. The vector from the center of table to the block
        # 2. The world y-axis for blue team and -y-axis for red team
        block_vector = np.array([x_world, y_world])
        world_vector = np.array([0, 1])
        if self.team == 'red':
            world_vector = -world_vector
        cos_theta = np.dot(block_vector, world_vector) / np.linalg.norm(block_vector)
        theta = np.arccos(cos_theta)

        # Reject blocks with angle less than the margin
        if theta < theta_margin:
            return False, None

        return True, theta


    """
    Control loop for dynamic pick and place task.
    Assume that you are starting from the start position and the gripper is open
    """
    def dynamic_pick_and_place(self):
        # Move to the safe position above the dynamic table area
        self.move_to_target(self.safe_dynamic_ee_pose_base)

        # Detect blocks
        detected_dynamic_blocks, detected_times = self.detect_blocks(validity_criteria='dynamic')

        # Choose one block to pick
        if len(detected_dynamic_blocks) > 0:
            block_name, block_pose = detected_dynamic_blocks[0]
            detected_time = detected_times[0]
            self.debug_print(f"Block {block_name} is chosen for dynamic pick and place.")

            if self.grasp_moving_block(block_name, block_pose, detected_time):
                # Move to the safe position above the dynamic table area
                self.move_to_target(self.safe_dynamic_ee_pose_base)

                # Move to the safe position above the tower-building area
                self.move_to_target(self.safe_tower_ee_pose_base)

                # Place the block
                if self.place_block(block_name):
                    self.placed_moving_blocks += 1
                    self.debug_print(f"Placed {self.placed_moving_blocks} moving blocks.")
        else:
            print("No valid dynamic blocks detected.")


    """
    The main control loop for the pick and place task.
    """
    def pick_and_place(self):
        order_of_operations = [
            'dynamic',
            'static', 'static', 
            'static', 'static',
        ]

        for i, operation in enumerate(order_of_operations):
            self.debug_print(f"Attempting operation {i+1} in mode: {operation}.")

            # Always first move to the start position after opening the gripper
            self.arm.open_gripper()
            self.arm.safe_move_to_position(self.start_position)

            # Check the mode
            if operation == 'static':
                self.static_pick_and_place()
            elif operation == 'dynamic':
                self.dynamic_pick_and_place()


if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("pick_and_place_node")

    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])
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
