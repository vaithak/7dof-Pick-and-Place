import numpy as np
from math import pi
import rospy
from core.interfaces import ObjectDetector, ArmController
from lib.IK_position_null import IK
from lib.calculateFK import FK

class PickAndPlace:
    def __init__(self):
        rospy.init_node('pick_and_place_node')  
        self.detector = ObjectDetector()
        self.arm = ArmController()
        self.ik_solver = IK()
        self.fk_solver = FK()
        
    def detect_blocks(self):
        """
        Detect blocks in the environment
        """

        self.arm.open_gripper()
        rospy.sleep(0.5)  
       # Initial position
        start_position = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])
        self.arm.safe_move_to_position(start_position)

        blocks = {}
        H_ee_camera = self.detector.get_H_ee_camera()
        if H_ee_camera is None:
            rospy.logwarn("Failed to get transform between end-effector and camera.")
            return blocks
        
        for (block_name, pose) in self.detector.get_detections():
            rospy.loginfo(f"Detected block '{block_name}' with pose: {pose}")
            # Transform block pose from camera frame to world frame using FK
            current_joint_angles = self.arm.get_positions()
            _, H_base_ee = self.fk_solver.forward(current_joint_angles)
            H_base_block = H_base_ee @ H_ee_camera @ pose  # Full transformation to world frame
            blocks[block_name] = H_base_block
            rospy.loginfo(f"Block '{block_name}' transformed to base frame: {H_base_block[:3, 3]}")
        return blocks
    
    def approach_block(self, block_pose):
        # Set orientation to make gripper point downwards
        downward_orientation = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        approach_pose = block_pose.copy()
        approach_pose[:3, :3] = downward_orientation
        # Approach 15 cm above the block
        approach_offset = np.array([0, 0, 0.15])
        approach_pose[:3, 3] += approach_offset
        return approach_pose
    
    def plan_ik_path(self, target_pose):
        current_joint_angles = self.arm.get_positions()
        solution, _, success, _ = self.ik_solver.inverse(target_pose, current_joint_angles, method='J_pseudo', alpha=0.5)
        if not success:
            rospy.logwarn("No valid IK solution found for target pose")
        return solution if success else None
    
    def move_to_target(self, target_pose):
        joint_angles = self.plan_ik_path(target_pose)
        if joint_angles is not None:
            success = self.arm.safe_move_to_position(joint_angles)
            if not success:
                rospy.logwarn("Failed to move to the target joint positions")
        else:
            rospy.logwarn("Failed to plan IK path to target pose")
    
    def grasp_block(self, block_pose):
        grasp_orientation = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        grasp_pose = block_pose.copy()
        grasp_pose[:3, :3] = grasp_orientation
        grasp_pose[:3, 3] = block_pose[:3, 3]  
        
        self.move_to_target(grasp_pose)  
        self.move_to_target(grasp_pose)

        self.arm.exec_gripper_cmd(pos=0.02, force=50)  
        
        # Verify grasp using gripper state
        gripper_state = self.arm.get_gripper_state()
        if gripper_state['force'][0] > 30:
            rospy.loginfo(f"Successfully grasped block at pose: {block_pose[:3, 3]}")
            return True
        else:
            rospy.logwarn("Failed to grasp block")
            return False

        # # Verify grasp using gripper state
        # gripper_state = self.arm.get_gripper_state()
        # if gripper_state['is_grasped']:
        #     rospy.loginfo(f"Successfully grasped block at pose: {block_pose[:3, 3]}")
        #     return True
        # else:
        #     rospy.logwarn("Failed to grasp block")
        #     return False
    
    def place_block(self, place_pose):
        approach_offset = np.array([0, 0, 0.15])  # Approach 15 cm above the placement
        approach_pose = place_pose.copy()
        approach_pose[:3, 3] += approach_offset
        self.move_to_target(approach_pose)
        
        self.move_to_target(place_pose)
        
        self.arm.exec_gripper_cmd(pos=0.08, force=10)
        rospy.loginfo(f"Block placed at pose: {place_pose[:3, 3]}")
        
        self.move_to_target(approach_pose)
    
    def pick_and_place(self, source_block, destination_pose):

        blocks = self.detect_blocks()
        rospy.loginfo(f"Available blocks: {list(blocks.keys())}")
        
        if source_block not in blocks:
            rospy.logwarn(f"Block {source_block} not detected")
            return False

        block_pose = blocks[source_block]
        
        if self.grasp_block(block_pose):
            self.place_block(destination_pose)
            return True
        
        rospy.logwarn(f"Failed to grasp block {source_block}")
        return False

def main():
    pick_place = PickAndPlace()
    source_block = "cube1_static"  
    destination_pose = np.eye(4)  # Replace with actual destination pose
    # destination_pose[:3, 3] = [1.159, -0.99, 0.25]
    
    success = pick_place.pick_and_place(source_block, destination_pose)
    
    if success:
        print("Pick and place completed successfully")
    else:
        print("Pick and place failed")

if __name__ == "__main__":
    main()
