#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        
        # Create publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Create subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        # Create timer for movement updates
        self.timer = self.create_timer(0.1, self.movement_callback)  # 10Hz
        
        # Movement parameters
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.5  # rad/s
        self.current_linear_speed = 0.0
        self.current_angular_speed = 0.0
        
        # Cleaning pattern parameters
        self.obstacle_threshold = 0.5  # meters
        self.living_being_threshold = 1.0  # meters
        self.min_scan_angle = -67.5  # degrees (135/2 degrees on each side)
        self.max_scan_angle = 67.5   # degrees
        
        # State variables
        self.is_turning = False
        self.turn_duration = 0.0
        self.turn_start_time = 0.0
        self.current_position = None
        self.current_orientation = None
        self.scan_data = None
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info('Planner node initialized')

    def odom_callback(self, msg):
        # Store current position and orientation
        self.current_position = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation

    def scan_callback(self, msg):
        # Store laser scan data
        self.scan_data = msg

    def check_for_obstacles(self):
        if self.scan_data is None:
            return False, False
        
        # Convert angle range to indices
        angle_increment = self.scan_data.angle_increment
        min_idx = int((self.min_scan_angle * math.pi / 180) / angle_increment)
        max_idx = int((self.max_scan_angle * math.pi / 180) / angle_increment)
        
        # Check for obstacles and living beings in the forward direction
        has_obstacle = False
        has_living_being = False
        
        for i in range(min_idx, max_idx):
            if i < 0 or i >= len(self.scan_data.ranges):
                continue
                
            distance = self.scan_data.ranges[i]
            
            # Skip invalid readings
            if distance == float('inf') or distance == 0.0:
                continue
                
            # Check for obstacles
            if distance < self.obstacle_threshold:
                has_obstacle = True
                break
                
            # Check for living beings (assuming they're detected at a greater distance)
            if distance < self.living_being_threshold:
                has_living_being = True
                break
        
        return has_obstacle, has_living_being

    def movement_callback(self):
        # Create Twist message
        twist = Twist()
        
        # Check for obstacles and living beings
        has_obstacle, has_living_being = self.check_for_obstacles()
        
        # If we're not currently turning, check if we need to turn
        if not self.is_turning:
            if has_obstacle or has_living_being:
                self.is_turning = True
                self.turn_duration = 1.0  # Turn for 1 second
                self.turn_start_time = self.get_clock().now().to_sec()
                # Turn right for obstacles, left for living beings
                self.current_angular_speed = -self.angular_speed if has_living_being else self.angular_speed
                self.current_linear_speed = 0.0
            else:
                self.current_linear_speed = self.linear_speed
                self.current_angular_speed = 0.0
        
        # If we're turning, check if we should stop
        elif self.is_turning:
            current_time = self.get_clock().now().to_sec()
            if current_time - self.turn_start_time >= self.turn_duration:
                self.is_turning = False
                self.current_linear_speed = self.linear_speed
                self.current_angular_speed = 0.0
        
        # Set the movement values
        twist.linear.x = self.current_linear_speed
        twist.angular.z = self.current_angular_speed
        
        # Publish the movement command
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    planner_node = PlannerNode()
    rclpy.spin(planner_node)
    planner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 