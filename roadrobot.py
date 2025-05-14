import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import LineString, Point, Polygon
from math import cos, sin, sqrt, pi, exp, atan2
from motion_planner import MotionPlanner

class Robot:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.orientation = random.uniform(0, 2 * pi)
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
        
    def set(self, new_x, new_y, new_orientation):
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation % (2 * pi))
        
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        self.forward_noise = float(new_f_noise)
        self.turn_noise = float(new_t_noise)
        self.sense_noise = float(new_s_noise)
    
    def copy(self):
        r = Robot()
        r.set(self.x, self.y, self.orientation)
        r.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return r
        
    def sense(self, nearby_landmarks):
        Z = []
        nearby_landmarks = list(nearby_landmarks)
        if len(nearby_landmarks) == 0:
            return Z  # No nearby landmarks

        for landmark in nearby_landmarks:
            dist = Point(self.x, self.y).distance(Point(landmark))# if landmark.dtype else Point(self.x, self.y).distance(Point(landmark.coords[0]))
            dist += random.gauss(0.0, self.sense_noise)
            Z.append(dist)
        
        return Z

    
    def move(self, turn, forward):
        if forward < 0:
            raise ValueError('Robot cannot move backwards')
        # Turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi
        
        # Move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + cos(orientation) * dist
        y = self.y + sin(orientation) * dist
        
        # Set particle
        res = Robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res
    
    def Gaussian(self, mu, sigma, x):
        if sigma == 0:
            sigma = 0.0001  # Prevent division by zero
        return exp(- ((mu - x) ** 2) / (2 * sigma ** 2)) / (sqrt(2.0 * pi) * sigma)
    
    def measurement_prob(self, measurement, landmarks):
        prob = 1.0
        
        # match landmarks to measurement length
        sensed_landmarks = landmarks[:len(measurement)]
        
        for landmark, z in zip(sensed_landmarks, measurement):
            dist = Point(self.x, self.y).distance(Point(landmark))
            prob *= self.Gaussian(dist, self.sense_noise, z)

        return prob


    
    def __repr__(self):
        return '[x=%.6f y=%.6f orient=%.6f]' % (self.x, self.y, self.orientation)