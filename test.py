from roadrobot import Robot
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import math
import numpy as np
import random
from shapely.geometry import LineString, Point, Polygon
from math import cos, sin, sqrt, pi, exp, atan2
from motion_planner import MotionPlanner
import os
import imageio
from imageio.plugins import freeimage
import matplotlib.pyplot as plt
import time

''''
Code for running and testing the particle filter
'''

#for saving particle filter steps as images
frame_folder = "frames"
os.makedirs(frame_folder, exist_ok=True)

random.seed(42)
np.random.seed(42)

location = "Tartu, Estonia"

# road network graph
G = ox.graph_from_place(location, network_type="drive")

# convert graph to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

# start and end points
# i chose Lille street -> Eha bus station
start = (58.3747761,26.7264392)
stop = (58.3697956,26.7278322)
# these are for out of bounds experiment
#start = (58.3430137,26.7370483)
#stop = (58.3430233,26.7333314)

# find the nearest nodes on the graph
start_node = ox.distance.nearest_nodes(G, X=start[1], Y=start[0])
end_node = ox.distance.nearest_nodes(G, X=stop[1], Y=stop[0])

# shortest path between start and end nodes
route = nx.shortest_path(G, start_node, end_node, weight='length')

route_edges = edges.loc[edges.index.isin(list(zip(route[:-1], route[1:])))]

# create a LineString from the route edges
route_line = route_edges.union_all()

# convert to a GeoDataFrame
route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs=edges.crs)

#fetch street names as landmarks in Tartu
#here highway: bus_stop gives all bus stop names
streetnames = ox.features_from_place(location, tags={'highway': True})

# streetnames['name'] can fetch info based on deteced street name
# after that streetnames['geometry'] gives info either POINT(x,y) (should be bus stop) or LINETRING(x y, x y)

# project to UTM for accurate distance measurements
utm_crs = ox.projection.project_gdf(edges).crs
utm_crs = ox.projection.project_gdf(route_gdf).crs

nodes = nodes.to_crs(utm_crs)
edges = edges.to_crs(utm_crs)
streetnames = streetnames.to_crs(utm_crs)
route_gdf = route_gdf.to_crs(utm_crs)

# extract route points
route_nodes = nodes.loc[route]
route_points = list(route_nodes['geometry'])
#for getting the difference between vehicles position and estimated position
def eval_estimate(estimated_pos, true_pos):
    dx = estimated_pos[0] - true_pos[0]
    dy = estimated_pos[1] - true_pos[1]
    return math.hypot(dx, dy)

def resample_particles(particles, weights):
    #resample particles based on their weights
    N = len(particles)
    new_particles = []
    index = int(random.random() * N)
    beta = 0.0
    mw = max(weights)
    for _ in range(N):
        beta += random.uniform(0, 2 * mw)
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % N
        new_particle = Robot()
        new_particle.set(particles[index].x, particles[index].y, particles[index].orientation)
        new_particle.set_noise(particles[index].forward_noise, particles[index].turn_noise, particles[index].sense_noise)
        new_particles.append(new_particle)
    return new_particles

def initialize_particles_near_roads(N, edges):
    #the particles need to be close to the road network
    #this improves performance and accuracy
    particles = []
    
    while len(particles) < N:
        # select a random road segment
        random_edge = edges.sample(n=1).iloc[0]
        
        # road geometry
        if isinstance(random_edge.geometry, LineString):
            road_line = random_edge.geometry
        else:
            continue
        
        # choose a random point along the road
        random_point = road_line.interpolate(random.uniform(0, road_line.length))

        s = random.uniform(0, road_line.length)
        pt1 = road_line.interpolate(s)
        pt2 = road_line.interpolate(min(road_line.length, s + 0.1))
        angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x)
        
        offset_distance = 3
        x = random_point.x + offset_distance * cos(angle + pi/2)  #offset to right side
        y = random_point.y + offset_distance * sin(angle + pi/2)

        # Add some noise
        x += random.gauss(0, 0.5)
        y += random.gauss(0, 0.5)
        
        # Set particle orientation similar to road direction
        orientation = angle + random.gauss(0, 0.1)

        # Create particle
        particle = Robot()
        particle.set(x, y, orientation)
        particle.set_noise(0.5, 0.3, 2.0)
        particles.append(particle)
    
    return particles

def get_closest_landmarks(robot, detected_name, isStation, closest_landmarks):
    #this function fetches the closest OSM landmarks based on 
    #ground truth landmarks defined in presetStreetnames

    if isStation:
        landmarks = streetnames['geometry'].loc[(streetnames['name'] == detected_name) & (streetnames['highway']=='bus_stop')]
    else:
        landmarks = streetnames['geometry'].loc[(streetnames['name'] == detected_name) & (streetnames['highway']!='bus_stop')]#fetch geometries based on name
    for streetname in landmarks:
        for sn in streetname.coords:
            closest_landmarks.append(sn)    
    nearby_landmarks = closest_landmarks

    if nearby_landmarks is None or len(nearby_landmarks) == 0:
        print("No landmarks in front of the car!")
        return gpd.GeoDataFrame(columns=streetnames.columns)

    print(f"Selected {len(nearby_landmarks)} closest landmarks.")
    return nearby_landmarks

def get_streetname(robot, nameindex, presetStreetnames, x_coord, y_coord):
    #this returns the street name for function get_closest_landmarks
    #based on how close is the vehicle to the ground truth landmark
    #or to avoid getting no street name at all, landmark which is closest to the predefined path is chosen
    if nameindex >= len(presetStreetnames):
        return ''
    location = presetStreetnames.iloc[nameindex]['geometry']
    diff_r = sqrt((robot.x - location.x)**2 + (robot.y - location.y)**2)
    diff_p = sqrt((x_coord - location.x)**2 + (y_coord - location.y)**2)
    if diff_r <= 150 or diff_p <=150:
        return presetStreetnames.iloc[nameindex]['name']
    else:
        return ''

# Filter out landmarks that are behind the car
# also landmarks that are too far away
def is_in_front(robot, landmark):
    if isinstance(landmark, Point):
        dx = landmark.x - robot.x
        dy = landmark.y - robot.y
    else:
        dx = landmark[0] - robot.x
        dy = landmark[1] - robot.y
    angle_to_landmark = atan2(dy, dx)
    angle_diff = (angle_to_landmark - robot.orientation + pi) % (2 * pi) - pi
    #return (abs(angle_diff) < pi / 2) #only in front [for testing]
    #return (sqrt(dx**2+dy**2) <=200)#only distance [for testing]
    return (abs(angle_diff) < pi / 2) & (sqrt(dx**2+dy**2) <=200)  # Only keep landmarks in front (±90 degrees) and within 200 meters

def snapToRoad(robot, edges):
    #place the robot on tho the closest road
    nearest_edge = edges.distance(Point(robot.x, robot.y)).idxmin()
    nearest_geom = edges.loc[nearest_edge].geometry
    nearest_point = nearest_geom.interpolate(nearest_geom.project(Point(robot.x, robot.y)))
    robot.set(nearest_point.x, nearest_point.y, robot.orientation)

def snapToRoad2(robot, edges, prev_edge_id=None, angle_weight=0.5, continue_weight=1.0):
    #place the robot on to the closest road and include previous road information as well
    point = Point(robot.x, robot.y)

    # nearby edges
    distances = edges.distance(point)
    candidates = distances.nsmallest(3).index  #top 3 closest roads

    best_score = -np.inf
    best_edge_id = None
    best_point = None

    for edge_id in candidates:
        edge = edges.loc[edge_id]
        geom = edge.geometry

        proj_dist = geom.project(point)
        snapped_point = geom.interpolate(proj_dist)

        dist = point.distance(snapped_point)
        closeness = -dist

        dx = geom.coords[1][0] - geom.coords[0][0]
        dy = geom.coords[1][1] - geom.coords[0][1]
        road_angle = atan2(dy, dx)

        angle_diff = abs((robot.orientation - road_angle + pi) % (2 * pi) - pi)
        direction_similarity = cos(angle_diff)  # 1 = aligned, -1 = opposite

        continue_bonus = 1.0 if edge_id == prev_edge_id else 0.0

        score = angle_weight * direction_similarity + continue_weight * continue_bonus + closeness

        if score > best_score:
            best_score = score
            best_edge_id = edge_id
            best_point = snapped_point

    #snap the robot to the best point
    robot.set(best_point.x, best_point.y, robot.orientation)
    return best_edge_id

def get_street_midpoint(landmarks, noise_std=1.0):
    #this function is used oonly once when initializing the vehicle
    #based on the first sensed landmark

    #convert to coordinates
    coords = [(pt.x, pt.y) for pt in landmarks if isinstance(pt, Point)]
    
    if len(coords) < 2:
        return coords[0]  #not enough points for a line
    
    line = LineString(coords)
    midpoint = line.interpolate(line.length / 2)

    #add some noise
    noisy_x = midpoint.x + random.gauss(0, noise_std)
    noisy_y = midpoint.y + random.gauss(0, noise_std)
    
    return noisy_x, noisy_y


route_tuples = [(p.x, p.y) for p in route_points]
# Initialize MotionPlanner with the waypoints
planner = MotionPlanner(route_tuples, 50)# moving in 50 meter intervals? can increase
motions = planner.get_motion_commands()  # Get planned turn-distance commands

# Initialize Robot
myrobot = Robot()
random_edge = edges.sample(n=1).iloc[0]
        
# road geometry
if isinstance(random_edge.geometry, LineString):
    road_line = random_edge.geometry
else:
    road_line = edges.sample(n=1).iloc[0].geometry

# choose a random point along the road
random_point = road_line.interpolate(random.uniform(0, road_line.length))
start_x, start_y = random_point.x, random_point.y  # random waypoint
myrobot.set(start_x, start_y, 0)  # Start facing 0 radians
#myrobot.set_noise(0.5, 0.1, 2.0)
myrobot.set_noise(0.3, 0.05, 1.0)

def evaluate_particle_filter(N, myrobot, particles, motions, presetStreetnames):
    #this is for evaluating the particle filter
    #errors, spreads, true_path and estimated_path are recorded for plotting
    errors = []
    spreads_x = []
    spreads_y = []
    estimated_path = []
    true_path = []
    prev_edge_id = None

    for t, (turn, distance, x_coord, y_coord) in enumerate(motions):

        closest_landmarks = []
        if t==0: #first one we need the given street sign just to start from somewhere
            detected_name = presetStreetnames.iloc[0]['name']#Lille
            isStation = presetStreetnames.iloc[0]['bus_stop']
            nameIndex = 0
            closest_landmarks.append(presetStreetnames.iloc[0]['geometry'])
            closest_landm = get_closest_landmarks(myrobot, detected_name, isStation, closest_landmarks)
            #Now we would like to chnge robot position according to that detected street
            mean_x, mean_y = get_street_midpoint(closest_landm)
            myrobot.set(mean_x, mean_y, turn)
            prev_edge_id = snapToRoad2(myrobot, edges, prev_edge_id)
            #snapToRoad(myrobot, edges)
        else:
            detected_name = get_streetname(myrobot, nameIndex, presetStreetnames, x_coord, y_coord) # need a function for retrieving the name based on how close it is(100 or 150 meters?)
            if detected_name != '':
                isStation = presetStreetnames.iloc[nameIndex]['bus_stop']
                closest_landmarks.append(presetStreetnames.iloc[nameIndex]['geometry'])
                nameIndex += 1
            # Move the robot using planner's motions
            myrobot = myrobot.move(turn, distance)
            prev_edge_id = snapToRoad2(myrobot, edges, prev_edge_id)
            #snapToRoad(myrobot, edges)
            if detected_name == '':
                closest_landm = []
            else:
                closest_landm = get_closest_landmarks(myrobot, detected_name, isStation, closest_landmarks)
                closest_landm = [landmark for landmark in closest_landm if is_in_front(myrobot, landmark)]# filter out behind robot
        Z = myrobot.sense(closest_landm) if not len(closest_landm) == 0 else []

        # Move particles
        particles = [p.move(turn, distance) for p in particles]

        # Calculate weights
        if len(closest_landm) == 0:
            #print("No landmarks detected, moving particles based on motion.")
            # giving uniform weights when no landmakrs detected
            weights = [1.0 / N] * N
        else:
            weights = [p.measurement_prob(Z, closest_landm) for p in particles]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / N] * N
        # Resample particles
        particles = resample_particles(particles, weights)

        # Estimate position
        x_est = np.mean([p.x for p in particles])
        y_est = np.mean([p.y for p in particles])
        estimated_path.append((x_est, y_est))

        # True position
        true_path.append((myrobot.x, myrobot.y))

        # Error and spread
        errors.append(eval_estimate((x_est, y_est), (myrobot.x, myrobot.y)))
        spreads_x.append(np.std([p.x for p in particles]))
        spreads_y.append(np.std([p.y for p in particles]))

    '''# Plot Error
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(errors, label="Localization error")
    plt.xlabel("Timestep")
    plt.ylabel("Error")
    plt.title("Error over time")
    plt.grid(True)
    plt.legend()

    # Plot Spread
    plt.subplot(1, 3, 2)
    plt.plot(spreads_x, label="X spread")
    plt.plot(spreads_y, label="Y spread")
    plt.xlabel("Timestep")
    plt.ylabel("Std dev")
    plt.title("Particle spread")
    plt.grid(True)
    plt.legend()

    # Plot Paths
    plt.subplot(1, 3, 3)
    true_x, true_y = zip(*true_path)
    est_x, est_y = zip(*estimated_path)
    plt.plot(true_x, true_y, label="True path", linewidth=2)
    plt.plot(est_x, est_y, '--', label="Estimated path", linewidth=2)
    if len(closest_landm) > 0:
        #closest_landmarks.append(closest_landm)
        coords = [
            (p.x, p.y) if hasattr(p, "x") else p
            for p in closest_landm
        ]
        plt.scatter(*zip(*coords), marker='*', c='blue', label='Landmarks')
    #plt.scatter([l[0] for l in landmarks], [l[1] for l in landmarks], marker='*', c='blue', label='Landmarks')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path comparison")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.tight_layout()
    plt.show()'''

    return errors, spreads_x, spreads_y

def compare_particle_counts(particle_amount, myrobot, motions, landmarks):
    #function to compare different particle counts
    avg_errors = []
    avtimes = []
    std_times = []
    std_errors = []

    for N in particle_amount:
        averrors = []
        times = []
        for i in range(20):
            # Initialize particles
            start = time.time()
            particles = initialize_particles_near_roads(N, edges)

            # Reset robot to same initial state
            r = myrobot.copy()

            # Run evaluation
            errors, _, _ = evaluate_particle_filter(N, r, particles, motions, landmarks)
            end = time.time()
            averrors.append(np.mean(errors))
            times.append(end-start)
        avtimes.append(np.mean(times))
        std_times.append(np.std(times))
        avg_errors.append(np.mean(averrors))
        std_errors.append(np.std(averrors))

        #print(f"Particles: {N}, Avg error: {avg_errors[-1]:.2f}")
    print("Errors:", avg_errors)
    print("std errors:", std_errors)
    print("Times:", avtimes)
    print("std times:", std_times)

    # Plot results

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.errorbar(particle_amount, avg_errors, yerr=std_errors, fmt='-o', capsize=5)
    plt.xlabel("Number of particles")
    plt.ylabel("Average localization error (m)")
    plt.title("Effect of particle count on accuracy")
    plt.grid(True)

    # Plot execution time
    plt.subplot(1, 2, 2)
    plt.errorbar(particle_amount, avtimes, yerr=std_times, fmt='-o', color='orange', capsize=5)
    plt.xlabel("Number of particles")
    plt.ylabel("Average execution time (s)")
    plt.title("Effect of particle count on execution time")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()




# Particle Filter Loop

def particle_filter(N, myrobot, motions, presetStreetnames):
    estimations = []
    true_path = []
    spreads_x = []
    spreads_y = []
    particles = initialize_particles_near_roads(N, edges)

    fig, ax = plt.subplots(figsize=(5, 10))
    xmin, ymin, xmax, ymax = route_gdf.total_bounds
    padding = 400
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)
    edges.plot(ax=ax, color="gray", linewidth=0.5, alpha=0.5, label="Roads")
    particle_positions = [(p.x, p.y) for p in particles]
    if particle_positions:
        plt.scatter(*zip(*particle_positions), color='red', s=5, alpha=0.5, label='Particles')
    plt.legend(loc='upper right', fontsize=8)    
    plt.title(f"Initialized particles")
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    frame_filename = os.path.join(frame_folder, f"frame_000.png")
    plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    prev_edge_id = None #this is for snapToRoad2 function
    for t, (turn, distance, x_coord, y_coord) in enumerate(motions):
        # we need x and y coords to accurately add street name detection where the car should see it
        print(f"Turn: {turn}, Didtance: {distance}")
        closest_landmarks = []
        if t==0: #first one we need the given street sign just to start from somewhere
            detected_name = presetStreetnames.iloc[0]['name']#Lille
            isStation = presetStreetnames.iloc[0]['bus_stop']
            nameIndex = 0
            closest_landmarks.append(presetStreetnames.iloc[0]['geometry'])
            closest_landm = get_closest_landmarks(myrobot, detected_name, isStation, closest_landmarks)
            #Now we would like to chnge robot position according to that detected street
            mean_x, mean_y = get_street_midpoint(closest_landm)
            myrobot.set(mean_x, mean_y, turn)
            prev_edge_id = snapToRoad2(myrobot, edges, prev_edge_id)
            #snapToRoad(myrobot, edges)
        else:
            detected_name = get_streetname(myrobot, nameIndex, presetStreetnames, x_coord, y_coord) # need a function for retrieving the name based on how close it is(100 or 150 meters?)
            if detected_name != '':
                isStation = presetStreetnames.iloc[nameIndex]['bus_stop']
                closest_landmarks.append(presetStreetnames.iloc[nameIndex]['geometry'])
                nameIndex += 1
            # Move the robot using planner's motions
            myrobot = myrobot.move(turn, distance)
            prev_edge_id = snapToRoad2(myrobot, edges, prev_edge_id)
            #snapToRoad(myrobot, edges)
            if detected_name == '':
                closest_landm = []
            else:
                closest_landm = get_closest_landmarks(myrobot, detected_name, isStation, closest_landmarks)
                closest_landm = [landmark for landmark in closest_landm if is_in_front(myrobot, landmark)]# filter out behind robot
        Z = myrobot.sense(closest_landm) if not len(closest_landm) == 0 else []

        # Move particles
        particles = [p.move(turn, distance) for p in particles]

        # Calculate weights
        if len(closest_landm) == 0:
            print("No landmarks detected, moving particles based on motion and assigning uniform weights.")
            # giving uniform weights when no landmakrs detected
            weights = [1.0 / N] * N
        else:
            weights = [p.measurement_prob(Z, closest_landm) for p in particles]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / N] * N
        # Resample particles
        particles = resample_particles(particles, weights)

        # Estimate position
        x_estimate = sum(p.x for p in particles) / N
        y_estimate = sum(p.y for p in particles) / N
        estimations.append((x_estimate, y_estimate))

        true_path.append((myrobot.x, myrobot.y))

        # Evaluate error
        error = eval_estimate((x_estimate, y_estimate), (myrobot.x, myrobot.y))
        errors.append(error)
        spreads_x.append(np.std([p.x for p in particles]))
        spreads_y.append(np.std([p.y for p in particles]))

        print(f"Step {t + 1}/{len(motions)}: Estimate = ({x_estimate:.2f}, {y_estimate:.2f}), "
            f"Actual = ({myrobot.x:.2f}, {myrobot.y:.2f}), Error = {error:.2f} meters")
        # Visualization

        fig, ax = plt.subplots(figsize=(5, 10))
        
        xmin, ymin, xmax, ymax = route_gdf.total_bounds
        padding = 400
        ax.set_xlim(xmin - padding, xmax + padding)
        ax.set_ylim(ymin - padding, ymax + padding)

        # Plot road network as background
        edges.plot(ax=ax, color="gray", linewidth=0.5, alpha=0.5, label="Roads")
        # Plot the planned route
        route_gdf.plot(ax=ax, color="blue", linewidth=2, label="Route")

        plt.scatter(preset_x, preset_y, color='orange', s=8, alpha=0.5, label='Street signs (ground truth)')


        # Plot elements
        if len(closest_landm) > 0:
            #closest_landmarks.append(closest_landm)
            coords = [
                (p.x, p.y) if hasattr(p, "x") else p
                for p in closest_landm
            ]
            plt.scatter(*zip(*coords), color='green', s=10, alpha=0.7, label="Landmarks (from OSM)")

        # Particles
        if particles:
            for p in particles:
                plt.arrow(p.x, p.y, 5 * math.cos(p.orientation), 5 * math.sin(p.orientation), 
                        head_width=3, color='red', alpha=0.3)

        # vehicle
        plt.scatter(myrobot.x, myrobot.y, color='purple', s=50, edgecolors='black', label='Vehicle (ground truth)')

        # estimation
        plt.scatter(x_estimate, y_estimate, color='yellow', s=30, edgecolors='black', label='Vehicle (estimated pos)')


        plt.legend(loc='upper right', fontsize=8)    
        plt.title(f"Particle filter at step {t + 1}")
        plt.xlabel('UTM Easting (meters)')
        plt.ylabel('UTM Northing (meters)')
        frame_filename = os.path.join(frame_folder, f"frame_{t+1:03d}.png")
        plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print("Average localization error:", sum(errors) / len(errors))
    '''fig, ax = plt.subplots(figsize=(5, 10))
    xmin, ymin, xmax, ymax = route_gdf.total_bounds
    padding = 400
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)
    edges.plot(ax=ax, color="gray", linewidth=0.5, alpha=0.5, label="Roads")
    true_x, true_y = zip(*true_path)
    est_x, est_y = zip(*estimations)
    plt.plot(true_x, true_y, label="True path", linewidth=2)
    plt.plot(est_x, est_y, '--', label="Estimated path", linewidth=2)
    plt.legend(loc='upper right', fontsize=8)    
    plt.title(f"Ground truth vs estimated path")
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.show()
    
    # Plot Error
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(errors, label="Localization error")
    plt.xlabel("Timestep")
    plt.ylabel("Error")
    plt.title("Error over time")
    plt.grid(True)
    plt.legend()

    # Plot Spread
    plt.subplot(1, 3, 2)
    plt.plot(spreads_x, label="X spread")
    plt.plot(spreads_y, label="Y spread")
    plt.xlabel("Timestep")
    plt.ylabel("Std dev")
    plt.title("Particle spread")
    plt.grid(True)
    plt.legend()

    # Plot Paths
    plt.subplot(1, 3, 3)
    true_x, true_y = zip(*true_path)
    est_x, est_y = zip(*estimations)
    plt.plot(true_x, true_y, label="True path", linewidth=2)
    plt.plot(est_x, est_y, '--', label="Estimated path", linewidth=2)
    if len(closest_landm) > 0:
        coords = [
            (p.x, p.y) if hasattr(p, "x") else p
            for p in closest_landm
        ]
        plt.scatter(*zip(*coords), marker='*', c='blue', label='Landmarks')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path comparison")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.tight_layout()
    plt.show()'''

# we have street name with according coordinate where the car sees/'detects' it 
# and whether its a bus stop or not[which is the boolean value]
#correct data
presetStreetnames = [['Lille', Point(26.7264392, 58.3747761), False], ['Lille', Point(26.7276707, 58.3750425), False],
                 ['Kalevi', Point(26.7293673, 58.3744809), False], ['Kalevi', Point(26.7311323, 58.3731457), False],
                 ['Kalevi', Point(26.7316183, 58.3727623), False], ['Kalevi', Point(26.7318824, 58.3725216), False],
                 ['Pargi', Point(26.7300876, 58.3721551), False], ['Pargi', Point(26.7284825, 58.3716853), False],
                 ['Tähe', Point(26.7273889, 58.370436), False],['Eha', Point(26.7277736, 58.3698797), True],
                 ['Tähe', Point(26.7278322, 58.3697956), False]]
#corrupted data, chnaged Kalevi to Turu or Võru
presetStreetnames1 = [['Lille', Point(26.7264392, 58.3747761), False], ['Lille', Point(26.7276707, 58.3750425), False],
                 ['Võru', Point(26.7293673, 58.3744809), False], ['Võru', Point(26.7311323, 58.3731457), False],
                 ['Võru', Point(26.7316183, 58.3727623), False], ['Kalevi', Point(26.7318824, 58.3725216), False],
                 ['Pargi', Point(26.7300876, 58.3721551), False], ['Pargi', Point(26.7284825, 58.3716853), False],
                 ['Tähe', Point(26.7273889, 58.370436), False],['Eha', Point(26.7277736, 58.3698797), True],
                 ['Tähe', Point(26.7278322, 58.3697956), False]]
#data for out-of-bounds experiment
presetStreetnames2 = [['Turu ring', Point(26.7370483, 58.3430137), True], ['Tähe', Point(26.7333314, 58.3430233), False],
                      ['Tähe', Point(26.7316678, 58.3430314), False], ['Tähe', Point(26.7312362, 58.341962), False],
                      ['Tähe', Point(26.7313399, 58.3408762), False]]

#give small noise to the recordings
for i, (name, geometry, bus_stop) in enumerate(presetStreetnames):
    theta = 0.0001
    presetStreetnames[i] = (name, Point(geometry.x+theta, geometry.y+theta), bus_stop)
#convert to correct format
presetStreetnames = gpd.GeoDataFrame(presetStreetnames, columns=['name', 'geometry', 'bus_stop'])
presetStreetnames.crs = 'epsg:4326'
presetStreetnames = presetStreetnames.to_crs(utm_crs)

presetStreetnames2 = gpd.GeoDataFrame(presetStreetnames2, columns=['name', 'geometry', 'bus_stop'])
presetStreetnames2.crs = 'epsg:4326'
presetStreetnames2 = presetStreetnames2.to_crs(utm_crs)

#for visualization purposes
preset_x = [x for x, y in zip(presetStreetnames.geometry.x, presetStreetnames.geometry.y)]
preset_y = [y for x, y in zip(presetStreetnames.geometry.x, presetStreetnames.geometry.y)]
errors = []


#evaluate_particle_filter(2500, myrobot, particles, motions, presetStreetnames.geometry)
#particle_amount = [500, 1000, 1250, 1500, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 4000]#for comparing particle counts
#compare_particle_counts(particle_amount, myrobot, motions, presetStreetnames)

#this is for testing outside of bounds case
#added artificial motions to keep the vehicle moving
'''motions[-1][-1]
motions.append((0.75, 50, motions[-1][-2], motions[-1][-1]))
for i in range(15):
    motions.append((0.0, 50, motions[-1][-2], motions[-1][-1]))
'''
#particle_amount = [2000]
#compare_particle_counts(particle_amount, myrobot, motions, presetStreetnames)

#the main testing part
start = time.time()
particle_filter(2000, myrobot, motions, presetStreetnames)#tried with 2000, 2500, 3000, 3500
end = time.time()
print("Execution time: ",end-start)
freeimage.download()
images = []
for file_name in sorted(os.listdir(frame_folder)):
    if file_name.endswith('.png'):
        file_path = os.path.join(frame_folder, file_name)
        images.append(imageio.v2.imread(file_path))

imageio.mimsave("particle_filter.gif", images, duration=10.0, plugin='pyav')