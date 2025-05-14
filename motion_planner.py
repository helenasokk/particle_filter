import numpy as np

class Waypoint:
    def __init__(self, x, y, speed=0.0, rotation=0.0):
        self.x = x
        self.y = y
        self.speed = speed  # Linear velocity
        self.rotation = rotation  # Rotation angle in radians

class MotionPlanner:
    def __init__(self, path, dist, max_speed=70.0, acceleration=0.5):
        self.path = path  # List of (x, y) tuples from the shortest path
        self.dist = dist
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.waypoints = self.generate_waypoints()
    
    def normalize_angle(self, angle):
        """Normalize angle to be within [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def generate_waypoints(self):
        MAX_TURN_ANGLE = np.radians(100)
        waypoints = []
        prev_x, prev_y = self.path[0]

        for i in range(1, len(self.path)):
            x, y = self.path[i]
            dx, dy = x - prev_x, y - prev_y
            distance = np.hypot(dx, dy)

            rotation = np.arctan2(dy, dx)
            '''if waypoints:
                prev_rotation = waypoints[-1].rotation
                delta_rotation = self.normalize_angle(rotation - prev_rotation)
                if abs(delta_rotation) > MAX_TURN_ANGLE:
                    print(f"Warning: Skipping sharp turn of {np.degrees(delta_rotation):.2f}° at ({x:.2f}, {y:.2f})")
                    continue'''

            #number of segments to divide the path into
            if distance > self.dist:
                num_segments = int(distance // self.dist)
                remainder = distance % self.dist

                ux, uy = dx / distance, dy / distance

                for j in range(1, num_segments + 1):
                    seg_x = prev_x + ux * self.dist * j
                    seg_y = prev_y + uy * self.dist * j
                    waypoints.append(Waypoint(seg_x, seg_y, self.dist, rotation))

                #add the last small segment (remainder)
                if remainder > 0:
                    seg_x = prev_x + ux * (self.dist * num_segments + remainder)
                    seg_y = prev_y + uy * (self.dist * num_segments + remainder)
                    waypoints.append(Waypoint(seg_x, seg_y, remainder, rotation))
            else:
                waypoints.append(Waypoint(x, y, distance, rotation))

            prev_x, prev_y = x, y

        return waypoints

    
    def get_motion_commands(self):
        """Returns (turn, distance) pairs from waypoints."""
        motions = []
        prev_rotation = 0.0  # Assume starting rotation is 0

        for wp in self.waypoints:
            turn = wp.rotation - prev_rotation  # Change in rotation
            distance = wp.speed  # Use the speed as the travel distance
            motions.append((turn, distance, wp.x, wp.y))
            prev_rotation = wp.rotation

        return motions

    def follow_path(self):
        """Simulate movement along the path"""
        for wp in self.waypoints:
            print(f"Moving to ({wp.x:.2f}, {wp.y:.2f}) | Speed: {wp.speed:.2f} m/s | Rotation: {np.degrees(wp.rotation):.2f}°")

if __name__ == "__main__":
    test_path = [(0, 0), (1, 1), (3, 2), (5, 3)]
    planner = MotionPlanner(test_path)
    planner.follow_path()