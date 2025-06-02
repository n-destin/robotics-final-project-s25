#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, PoseStamped
import math
from nav_msgs.msg import Odometry, Path
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
        
        # Create publisher for path visualization
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        
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
        
        # Create timer for path publishing
        self.path_timer = self.create_timer(1.0, self.publish_path)  # 1Hz
        
        # Movement parameters
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
        
        # Get parameters from launch file or use defaults
        self.declare_parameter('grid_width', 3)
        self.declare_parameter('grid_height', 3)
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_speed', 0.5)
        
        self.grid_width = self.get_parameter('grid_width').value
        self.grid_height = self.get_parameter('grid_height').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        
        self.obstacles = []  # List of obstacle coordinates
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize path planning for Chinese Postman Algorithm
        self.initialize_path_planning()
        
        self.get_logger().info('Planner node initialized')

    def create_grid_graph(self, width, height, obstacles=None):
        """Create a grid graph representation of the space"""
        if obstacles is None:
            obstacles = []
            
        # Clear existing graph
        self.graph.clear()
            
        # Create nodes for each cell
        for x in range(width):
            for y in range(height):
                if (x, y) not in obstacles:
                    self.graph.add_node((x, y))
        
        # If no nodes were added, return
        if not self.graph.nodes():
            self.get_logger().warn('No valid nodes in graph')
            return
            
        # Create edges between adjacent cells
        for x in range(width):
            for y in range(height):
                if (x, y) in obstacles:
                    continue
                    
                # Check right neighbor
                if x + 1 < width and (x + 1, y) not in obstacles:
                    self.graph.add_edge((x, y), (x + 1, y), weight=1)
                
                # Check top neighbor
                if y + 1 < height and (x, y + 1) not in obstacles:
                    self.graph.add_edge((x, y), (x, y + 1), weight=1)
                
                # Check diagonal neighbors
                if x + 1 < width and y + 1 < height and (x + 1, y + 1) not in obstacles:
                    self.graph.add_edge((x, y), (x + 1, y + 1), weight=1.414)  # sqrt(2)
                if x + 1 < width and y - 1 >= 0 and (x + 1, y - 1) not in obstacles:
                    self.graph.add_edge((x, y), (x + 1, y - 1), weight=1.414)  # sqrt(2)
        
        # If no edges were added, add a self-loop to make the graph Eulerian
        if not self.graph.edges():
            if len(self.graph.nodes()) == 1:
                node = list(self.graph.nodes())[0]
                self.graph.add_edge(node, node, weight=0)
                self.get_logger().info('Added self-loop to single node')
            else:
                # Connect all nodes in a cycle
                nodes = list(self.graph.nodes())
                for i in range(len(nodes)):
                    v1 = nodes[i]
                    v2 = nodes[(i + 1) % len(nodes)]
                    weight = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
                    self.graph.add_edge(v1, v2, weight=weight)
                self.get_logger().info('Connected nodes in a cycle')
            return
            
        # Ensure the graph is connected
        if not nx.is_connected(self.graph):
            # Find connected components
            components = list(nx.connected_components(self.graph))
            # Connect components with minimum weight edges
            for i in range(len(components) - 1):
                comp1 = components[i]
                comp2 = components[i + 1]
                min_dist = float('inf')
                min_edge = None
                for v1 in comp1:
                    for v2 in comp2:
                        dist = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            min_edge = (v1, v2)
                if min_edge:
                    self.graph.add_edge(min_edge[0], min_edge[1], weight=min_dist)
        
        self.get_logger().info(f'Created graph with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges')
    
    def find_odd_degree_vertices(self):
        """Find all vertices with odd degree in the graph"""
        odd_vertices = [v for v in self.graph.nodes() if self.graph.degree(v) % 2 != 0]
        self.get_logger().info(f'Found {len(odd_vertices)} odd degree vertices')
        return odd_vertices
    
    def find_minimum_weight_matching(self, odd_vertices):
        """Find minimum weight perfect matching for odd degree vertices"""
        if not odd_vertices:
            return set()
            
        matching_graph = nx.Graph()
        
        # Add edges between all pairs of odd vertices
        for i in range(len(odd_vertices)):
            for j in range(i + 1, len(odd_vertices)):
                v1, v2 = odd_vertices[i], odd_vertices[j]
                # Use Euclidean distance as weight
                weight = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
                matching_graph.add_edge(v1, v2, weight=weight)
        
        # Find minimum weight matching
        try:
            matching = nx.min_weight_matching(matching_graph)
            self.get_logger().info(f'Found {len(matching)} matching pairs')
            return matching
        except nx.NetworkXError as e:
            self.get_logger().error(f'Error finding matching: {str(e)}')
            return set()
    
    def find_eulerian_circuit(self):
        """Find Eulerian circuit in the graph"""
        if not nx.is_connected(self.graph):
            self.get_logger().warn('Graph is not connected')
            return None
            
        try:
            # Check if graph is Eulerian
            if not nx.is_eulerian(self.graph):
                self.get_logger().warn('Graph is not Eulerian')
                return None
                
            # Find Eulerian circuit
            circuit = list(nx.eulerian_circuit(self.graph))
            self.get_logger().info(f'Found Eulerian circuit with {len(circuit)} edges')
            return circuit
        except nx.NetworkXError as e:
            self.get_logger().error(f'Error finding Eulerian circuit: {str(e)}')
            return None
    
    def chinese_postman(self):
        """Solve the Chinese Postman Problem"""
        self.get_logger().info('Starting Chinese Postman algorithm')
        
        # Check if graph is empty
        if not self.graph.nodes():
            self.get_logger().warn('Graph is empty')
            return None
            
        # Check if graph is connected
        if not nx.is_connected(self.graph):
            self.get_logger().warn('Graph is not connected')
            return None
            
        # Find odd degree vertices
        odd_vertices = self.find_odd_degree_vertices()
        
        # If no odd vertices, graph is already Eulerian
        if not odd_vertices:
            self.get_logger().info('No odd vertices, graph is already Eulerian')
            circuit = self.find_eulerian_circuit()
            if circuit:
                return [edge[0] for edge in circuit] + [circuit[-1][1]]
            return None
        
        # Create a MultiGraph copy for modification (allows duplicate edges)
        working_graph = nx.MultiGraph(self.graph)
        self.get_logger().info(f'Created MultiGraph with {len(working_graph.nodes())} nodes and {len(working_graph.edges())} edges')
        
        # Find minimum weight matching
        matching = self.find_minimum_weight_matching(odd_vertices)
        self.get_logger().info(f'Processing {len(matching)} matching pairs')
        
        # Add matching edges to the working graph using shortest paths
        for v1, v2 in matching:
            try:
                # Find shortest path between matched vertices
                path = nx.shortest_path(self.graph, v1, v2, weight='weight')
                self.get_logger().info(f'Adding path from {v1} to {v2} with {len(path)-1} edges')
                self.get_logger().info(f'Path: {path}')
                
                # Check degrees before adding
                deg_v1_before = working_graph.degree(v1)
                deg_v2_before = working_graph.degree(v2)
                
                # Add edges along the path (duplicate existing edges in MultiGraph)
                for i in range(len(path) - 1):
                    # Get weight from original graph or calculate
                    if self.graph.has_edge(path[i], path[i + 1]):
                        weight = self.graph[path[i]][path[i + 1]]['weight']
                    else:
                        # Calculate weight for new edge
                        weight = math.sqrt((path[i][0] - path[i + 1][0])**2 + 
                                         (path[i][1] - path[i + 1][1])**2)
                    # Add edge to MultiGraph (will create duplicate)
                    working_graph.add_edge(path[i], path[i + 1], weight=weight)
                
                # Check degrees after adding
                deg_v1_after = working_graph.degree(v1)
                deg_v2_after = working_graph.degree(v2)
                self.get_logger().info(f'Degrees: {v1}: {deg_v1_before}->{deg_v1_after}, {v2}: {deg_v2_before}->{deg_v2_after}')
                
            except nx.NetworkXNoPath:
                self.get_logger().warn(f'No path found between {v1} and {v2}')
                continue
        
        self.get_logger().info('Added matching paths to working graph')
        
        # Check vertex degrees after matching
        remaining_odd = [v for v in working_graph.nodes() if working_graph.degree(v) % 2 != 0]
        if remaining_odd:
            self.get_logger().warn(f'Still have {len(remaining_odd)} odd vertices after matching: {remaining_odd}')
            for v in remaining_odd:
                self.get_logger().info(f'Vertex {v} has degree {working_graph.degree(v)}')
            return None
        else:
            self.get_logger().info('All vertices now have even degree!')
        
        self.get_logger().info('All vertices now have even degree - finding Eulerian circuit')
        
        # Find Eulerian circuit in the modified graph
        try:
            circuit = list(nx.eulerian_circuit(working_graph))
            if not circuit:
                self.get_logger().warn('Empty circuit found')
                return None
            self.get_logger().info(f'Found Eulerian circuit with {len(circuit)} edges')
            return [edge[0] for edge in circuit] + [circuit[-1][1]]
        except nx.NetworkXError as e:
            self.get_logger().error(f'Error finding Eulerian circuit: {str(e)}')
            return None
    
    def get_path_coordinates(self, path):
        """Convert grid coordinates to real-world coordinates"""
        return [(x * self.cell_size, y * self.cell_size) for x, y in path]

    def initialize_path_planning(self):
        """Initialize the path planning with the Chinese Postman algorithm"""
        # Create grid graph
        self.create_grid_graph(self.grid_width, self.grid_height, self.obstacles)
        
        # Check if all cells are obstacles
        if len(self.obstacles) == self.grid_width * self.grid_height:
            self.get_logger().warn('All cells are obstacles')
            self.current_path = None
            return
            
        # Check if graph is empty
        if not self.graph.nodes():
            self.get_logger().warn('Graph is empty after obstacle removal')
            self.current_path = None
            return
            
        # Check if graph is connected
        if not nx.is_connected(self.graph):
            self.get_logger().warn('Graph is not connected')
            self.current_path = None
            return
            
        # Check if graph has edges
        if not self.graph.edges():
            self.get_logger().warn('Graph has no edges')
            self.current_path = None
            return
            
        # Solve for optimal path
        path = self.chinese_postman()
        if path and len(path) > 1:  # Ensure we have at least 2 points for a valid path
            self.current_path = self.get_path_coordinates(path)
            self.current_path_index = 0
            self.get_logger().info(f'Path planning initialized successfully with {len(self.current_path)} points')
        else:
            self.get_logger().error('Failed to find valid path')
            self.current_path = None

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

    def publish_path(self):
        """Publish the planned path for visualization"""
        if not self.current_path:
            self.get_logger().warn('No current path to publish - path is None')
            return
            
        self.get_logger().info(f'Publishing path with {len(self.current_path)} points')
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'
        
        for i, (x, y) in enumerate(self.current_path):
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = 'odom'
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)
        
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    planner_node = PlannerNode()
    rclpy.spin(planner_node)
    planner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
