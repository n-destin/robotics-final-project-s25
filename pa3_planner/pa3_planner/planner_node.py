#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import networkx as nx
import numpy as np

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
        
        # Path planning variables
        self.graph = nx.Graph()
        self.cell_size = 1.0  # Size of each cell in meters
        self.current_path = None
        self.current_path_index = 0
        self.grid_width = 10  # Number of cells in width
        self.grid_height = 10  # Number of cells in height
        self.obstacles = []  # List of obstacle coordinates
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize path planning for Chinese Postman Algorithm
        self.initialize_path_planning()
        
        self.get_logger().info('Planner node initialized')

    # Create a grid graph representation of the space/map
    def create_grid_graph(self, width, height, obstacles=None):
        if obstacles is None:
            obstacles = []
            
        # Create nodes for each cell
        for x in range(width):
            for y in range(height):
                if (x, y) not in obstacles:
                    self.graph.add_node((x, y))
        
        # Create edges between adjacent cells
        # in order to give the robot the ability to traverse cells
        for x in range(width):
            for y in range(height):
                if (x, y) in obstacles:
                    continue
                    
                # Check right neighbor if not an obstacle
                if x + 1 < width and (x + 1, y) not in obstacles:
                    self.graph.add_edge((x, y), (x + 1, y), weight=1)
                
                # Check top neighbor if not an obstacle
                if y + 1 < height and (x, y + 1) not in obstacles:
                    self.graph.add_edge((x, y), (x, y + 1), weight=1)
    
    # Find all vertices with odd degree in the graph
    def find_odd_degree_vertices(self):
        return [v for v in self.graph.nodes() if self.graph.degree(v) % 2 != 0]
    
    # Find minimum weight perfect matching for odd degree vertices
    def find_minimum_weight_matching(self, odd_vertices):
        matching_graph = nx.Graph()
        
        # Add edges between all pairs of odd vertices
        for i in range(len(odd_vertices)):
            for j in range(i + 1, len(odd_vertices)):
                v1, v2 = odd_vertices[i], odd_vertices[j]
                weight = abs(v1[0] - v2[0]) + abs(v1[1] - v2[1])
                matching_graph.add_edge(v1, v2, weight=weight)
        
        matching = nx.max_weight_matching(matching_graph, maxcardinality=True)
        return matching
    
    # Find Eulerian circuit in the graph
    def find_eulerian_circuit(self):
        try:
            circuit = list(nx.eulerian_circuit(self.graph))
            return circuit
        except nx.NetworkXError:
            return None
    
    def chinese_postman(self):
        # Algorthim Implementation from: https://webspace.maths.qmul.ac.uk/b.jackson/MAS210/ch8.pdf
        odd_vertices = self.find_odd_degree_vertices()
        
        if not odd_vertices:
            circuit = self.find_eulerian_circuit()
            if circuit:
                return [edge[0] for edge in circuit] + [circuit[-1][1]]
        
        matching = self.find_minimum_weight_matching(odd_vertices)
        
        for v1, v2 in matching:
            path = nx.shortest_path(self.graph, v1, v2)
            for i in range(len(path) - 1):
                self.graph.add_edge(path[i], path[i + 1])
        
        circuit = self.find_eulerian_circuit()
        if circuit:
            return [edge[0] for edge in circuit] + [circuit[-1][1]]
        
        return None
    
    def get_path_coordinates(self, path):
        # Convert grid coordinates to real-world coordinates
        return [(x * self.cell_size, y * self.cell_size) for x, y in path]

    def initialize_path_planning(self):
        # Initialize the path planning with the Chinese Postman algorithm
        # Create grid graph
        self.create_grid_graph(self.grid_width, self.grid_height, self.obstacles)
        
        # Solve for optimal path
        path = self.chinese_postman()
        if path:
            self.current_path = self.get_path_coordinates(path)
            self.current_path_index = 0
            self.get_logger().info('Path planning initialized successfully')
        else:
            self.get_logger().error('Failed to find valid path')

    def get_angle_to_target(self, target_x, target_y):
        # Calculate the angle to the target point
        if self.current_position is None or self.current_orientation is None:
            return 0.0
            
        # Calculate angle to target
        dx = target_x - self.current_position.x
        dy = target_y - self.current_position.y
        target_angle = math.atan2(dy, dx)
        
        # Get current orientation
        current_angle = 2 * math.atan2(self.current_orientation.z, self.current_orientation.w)
        
        # Calculate angle difference
        angle_diff = target_angle - current_angle
        
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        return angle_diff

    def get_distance_to_target(self, target_x, target_y):
        # Calculate the distance to the target point
        if self.current_position is None:
            return float('inf')
            
        dx = target_x - self.current_position.x
        dy = target_y - self.current_position.y
        return math.sqrt(dx*dx + dy*dy)

    def odom_callback(self, msg):
        # Store current position and orientation
        self.current_position = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation

    def scan_callback(self, msg):
        # Store laser scan data
        self.scan_data = msg

    def check_for_obstacles(self):
        # Check for obstacles and living beings in the robot's path
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
                
            # Check for living beings
            if distance < self.living_being_threshold:
                has_living_being = True
                break
        
        return has_obstacle, has_living_being

    def movement_callback(self):
        # Handle robot movement based on the planned path
        if not self.current_path or self.current_path_index >= len(self.current_path):
            return
            
        # Create Twist message
        twist = Twist()
        
        # Get current target
        target_x, target_y = self.current_path[self.current_path_index]
        
        # Check for obstacles and living beings
        has_obstacle, has_living_being = self.check_for_obstacles()
        
        if has_obstacle or has_living_being:
            # Stop and turn
            self.is_turning = True
            self.turn_duration = 1.0
            self.turn_start_time = self.get_clock().now().to_sec()
            self.current_angular_speed = -self.angular_speed if has_living_being else self.angular_speed
            self.current_linear_speed = 0.0
        else:
            # Calculate angle and distance to target
            angle_to_target = self.get_angle_to_target(target_x, target_y)
            distance_to_target = self.get_distance_to_target(target_x, target_y)
            
            # If we're close enough to the target, move to next point
            if distance_to_target < 0.1:  # 10cm threshold
                self.current_path_index += 1
                return
                
            # If we need to turn, do that first
            if abs(angle_to_target) > 0.1:  # 0.1 rad threshold
                self.current_angular_speed = self.angular_speed if angle_to_target > 0 else -self.angular_speed
                self.current_linear_speed = 0.0
            else:
                # Move towards target
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