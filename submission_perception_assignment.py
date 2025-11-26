#!/usr/bin/env python
# coding: utf-8

# # Anerudh Raina's submission for Perception Assignment by 10xconstruction.ai

# ## Import required libraries

# In[69]:


import numpy as np
from collections import defaultdict
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import csv
from sensor_msgs.msg import Image
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn import linear_model
from scipy.spatial import ConvexHull
from numpy.linalg import norm
import math
import csv


# ## Parse rosbag file for messages

# In[5]:


def connect(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    return conn, c

def close(conn):
    conn.close()

def countRows(cursor, table_name, print_out=False):
    """ Returns the total number of rows in the database. """
    cursor.execute('SELECT COUNT(*) FROM {}'.format(table_name))
    count = cursor.fetchall()
    if print_out:
        print('\nTotal rows: {}'.format(count[0][0]))
    return count[0][0]

def getHeaders(cursor, table_name, print_out=False):
    """ Returns a list of tuples with column informations:
    (id, name, type, notnull, default_value, primary_key)
    """
    # Get headers from table "table_name"
    cursor.execute('PRAGMA TABLE_INFO({})'.format(table_name))
    info = cursor.fetchall()
    if print_out:
        print("\nColumn Info:\nID, Name, Type, NotNull, DefaultVal, PrimaryKey")
        for col in info:
            print(col)
    return info

def getAllElements(cursor, table_name, print_out=False):
    """ Returns a dictionary with all elements of the table database.
    """
    # Get elements from table "table_name"
    cursor.execute('SELECT * from({})'.format(table_name))
    records = cursor.fetchall()
    if print_out:
        print("\nAll elements:")
        for row in records:
            print(row)
    return records

def isTopic(cursor, topic_name, print_out=False):
    """ Returns topic_name header if it exists. If it doesn't, returns empty.
        It returns the last topic found with this name.
    """
    boolIsTopic = False
    topicFound = []

    # Get all records for 'topics'
    records = getAllElements(cursor, 'topics', print_out=False)

    # Look for specific 'topic_name' in 'records'
    for row in records:
        if(row[1] == topic_name): # 1 is 'name' TODO
            boolIsTopic = True
            topicFound = row
    if print_out:
        if boolIsTopic:
             # 1 is 'name', 0 is 'id' TODO
            print('\nTopic named', topicFound[1], ' exists at id ', topicFound[0] ,'\n')
        else:
            print('\nTopic', topic_name ,'could not be found. \n')

    return topicFound

def getAllMessagesInTopic(cursor, topic_name, print_out=False):
    """ Returns all timestamps and messages at that topic.
    There is no deserialization for the BLOB data.
    """
    count = 0
    timestamps = []
    messages = []

    # Find if topic exists and its id
    topicFound = isTopic(cursor, topic_name, print_out=False)

    # If not find return empty
    if not topicFound:
        print('Topic', topic_name ,'could not be found. \n')
    else:
        records = getAllElements(cursor, 'messages', print_out=False)

        # Look for message with the same id from the topic
        for row in records:
            if row[1] == topicFound[0]:     # 1 and 0 is 'topic_id' TODO
                count = count + 1           # count messages for this topic
                timestamps.append(row[2])   # 2 is for timestamp TODO
                messages.append(row[3])     # 3 is for all messages

        # Print
        if print_out:
            print('\nThere are ', count, 'messages in ', topicFound[1])

    return timestamps, messages

def getAllTopicsNames(cursor, print_out=False):
    """ Returns all topics names.
    """
    topicNames = []
    # Get all records for 'topics'
    records = getAllElements(cursor, 'topics', print_out=False)

    # Save all topics names
    for row in records:
        topicNames.append(row[1])  # 1 is for topic name TODO
    if print_out:
        print('\nTopics names are:')
        print(topicNames)

    return topicNames

def getAllMsgsTypes(cursor, print_out=False):
    """ Returns all messages types.
    """
    msgsTypes = []
    # Get all records for 'topics'
    records = getAllElements(cursor, 'topics', print_out=False)

    # Save all message types
    for row in records:
        msgsTypes.append(row[2])  # 2 is for message type TODO
    if print_out:
        print('\nMessages types are:')
        print(msgsTypes)

    return msgsTypes

def getMsgType(cursor, topic_name, print_out=False):
    """ Returns the message type of that specific topic.
    """
    msg_type = []
    # Get all topics names and all message types
    topic_names = getAllTopicsNames(cursor, print_out=False)
    msgs_types = getAllMsgsTypes(cursor, print_out=False)

    # look for topic at the topic_names list, and find its index
    for index, element in enumerate(topic_names):
        if element == topic_name:
            msg_type = msgs_types[index]
    if print_out:
        print('\nMessage type in', topic_name, 'is', msg_type)

    return msg_type

# path to the bagfile
bag_file = 'depth.db3'

# topic name
topic_name = '/depth'

### connect to the database
conn, c = connect(bag_file)

### get all topics names and types
topic_names = getAllTopicsNames(c, print_out=False)
topic_types = getAllMsgsTypes(c, print_out=False)

# Create a map for quicker lookup
type_map = {topic_names[i]:topic_types[i] for i in range(len(topic_types))}

### get all timestamps and all messages
# t is used as an array of timestamps throughout the code 
t, msgs = getAllMessagesInTopic(c, topic_name, print_out=True)

# Deserialize the message
msg_type = get_message(type_map[topic_name])  # Assuming type_map is a dictionary mapping topic names to message types

### close connection to the database
close(conn)


# ## Create dictionary of depth images indexed by the timestamp

# In[6]:


t_img = defaultdict()
for timestamp, message in zip(t,msgs):
    msg = deserialize_message(message, msg_type)
    np_arr = np.frombuffer(msg.data, dtype=np.uint16)
    t_img[timestamp] = np_arr.reshape(msg.height, msg.width)


# ## Create 3D points from depth map

# In[10]:


def depth_to_3d_points(depth_map, fx=1, fy=1, cx=0, cy=0):
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = h//2, w//2
    # Convert to float for calculations
    depth_map = depth_map.astype(np.float32)

    # Handle potential zero depth values (set to a small non-zero value or filter out)
    depth_map[depth_map == 0] = np.nan # Or a small value like 0.001

    # Handle large depth values (set to a small non-zero value or filter out)
    depth_map[depth_map >= 5000] = np.nan 

    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    # Stack the coordinates to get (N, 3) array of 3D points
    points_3d = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Filter out invalid points (e.g., from zero depth or nans)
    points_3d = points_3d[~np.isnan(points_3d).any(axis=1)]
    return points_3d


# In[11]:


# Create a dictionary of 3D points indexed by the timestamp of the frame
t_pt = defaultdict()
for i in range(len(t)):
    t_pt[t[i]] = depth_to_3d_points(t_img[t[i]])


# ## Extract the planes from the 3D points of the scene using RANSAC

# In[12]:


def find_planes_ransac(points_3d, min_samples=7000, residual_threshold=80.0, max_trials=100):
    planes = []
    remaining_points = points_3d.copy()

    while len(remaining_points) > min_samples:
        model = linear_model.RANSACRegressor(linear_model.LinearRegression(), min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials)

        # Fit a plane (z = ax + by + c)
        try:
            model.fit(remaining_points[:, :2], remaining_points[:, 2])
        except ValueError: # Not enough inliers
            break

        inlier_mask = model.inlier_mask_

        if np.sum(inlier_mask) < min_samples:
            break

        plane_points = remaining_points[inlier_mask]
        planes.append(plane_points)

        remaining_points = remaining_points[~inlier_mask]

    return planes


# In[13]:


# Create dictionary of planes indexed by the timestamps
t_pl = defaultdict()
for i in range(len(t)):
    t_pl[t[i]] = find_planes_ransac(t_pt[t[i]])


# ## Extract normal vector and visible area of the plane with the greatest area 

# In[61]:


# Define a function to return the normal vector and area of the plane using SVD and Convex Hull
def calculate_plane_properties(plane_points):
    if len(plane_points) < 3:
        return None, None

    # Randomly select 10000 points from the plane for efficient 
    # SVD calculation; this is an approximation
    num_rows_to_select = 10000
    shuffled_indices = np.random.permutation(plane_points.shape[0])
    selected_points = plane_points[shuffled_indices[:num_rows_to_select]]

    # Orientation (Normal Vector)
    # Fit a plane to the points: ax + by + cz = d
    # This can be done by SVD on the centered points
    centroid = np.mean(plane_points, axis=0)
    centered_points = selected_points - centroid
    _, _, V = np.linalg.svd(centered_points)
    normal_vector = V[2, :] # The last column of V is the normal vector

    # Ensure normal points consistently (towards the depth camera, inwards along -Z)
    if normal_vector[2] > 0:
        normal_vector = -normal_vector

    # Surface Area (Approximation using convex hull in 2D projection)
    # Project points onto a 2D plane perpendicular to the normal
    # This is a simplification; a more accurate area would consider 3D geometry
    # For a quick approximation, project onto the dominant 2D plane
    if abs(normal_vector[2]) > 0.5: # Mostly horizontal plane
        projected_points = plane_points[:, :2]
    elif abs(normal_vector[1]) > 0.5: # Mostly vertical along Y
        projected_points = plane_points[:, [0, 2]]
    else: # Mostly vertical along X
        projected_points = plane_points[:, [1, 2]]

    if len(projected_points) >= 3:
        hull = ConvexHull(projected_points)
        surface_area = hull.area
    else:
        surface_area = 0

    return surface_area, normal_vector/norm(normal_vector)


# In[62]:


# Use planes from the timestamps to retrieve normal vector of the plane with the greatest area
# and create a dictionary of the retrieved normal vectors indexed with the respective timestamps
# Since the task description mentions all units are in SI units, the depth map is presumed to 
# represent depth in metres
t_nv = defaultdict()
t_va = defaultdict()

for i in range(len(t)):
    plane_data = []
    for plane_points in t_pl[t[i]]:
        area, normal = calculate_plane_properties(plane_points)
        if area is not None:
            plane_data.append({'area': area, 'normal': normal, 'points': plane_points})

    # Sort the planes by area
    sorted_planes = sorted(plane_data, key=lambda x: x['area'], reverse=True)

    # Assign the plane with the highest area to the dictionary
    t_nv[t[i]] = sorted_planes[0]['normal']
    t_va[t[i]] = sorted_planes[0]['area']


# ## Calculate the angle between the normal vector of the plane with the greatest area and the camera normal

# In[63]:


# Define a function to calculate the angle, in radians, between the normal vector and the camera normal
def calculate_angle_to_camera_normal(plane_vector):
    # Set the camera normal following the illustration in the project description
    camera_normal = np.array([0, 0, -1])

    # Compute cosine similarity
    cosine = np.dot(plane_vector, camera_normal) / (norm(plane_vector) * norm(camera_normal))

    # Compute the angle
    angle = math.acos(cosine)

    return angle


# In[64]:


# Create a dictionary of the normal angle indexed with the timestamp from the normal vectors
t_na = defaultdict()
for i in range(len(t)):
    t_na[t[i]] = calculate_angle_to_camera_normal(t_nv[t[i]])


# ## Calculate angular velocity and axis ot rotation and put them into dictionaries indexed by the timestamps between contiguous frames

# In[65]:


# Create dictionary of the rotation axis indexed by the respective timestamp
t_rotax = defaultdict()

for i in range(1,len(t)):
    # Calculate the cross product of contiguous normal vectors, which gives us the axis of rotation
    axis = np.cross(t_nv[t[i-1]], t_nv[t[i]])
    t_rotax[t[i]] = axis


# In[66]:


# Print axis of rotation for the contiguous timestamps
for i in range(1,len(t)):
    print('Axis of rotation is: '+ str( t_rotax[t[i]]) + ', between timestamp: '+str(t[i-1])+' and '+str(t[i]))


# In[67]:


# Print normal vector and area of the plane with the greatest area for the timestamp and frame number
for i in range(len(t)):
    print('For frame number: '+str(i)+', normal vector is:'+str(t_nv[t[i]])+',\nvisible area: '+str(t_va[t[i]])+' sq.m., for the timestamp: '+str(t[i]))


# ## Create the table and text file and write to them

# In[68]:


# Open the CSV file for writing table normal angle and visible area
csv_file_path = 'table_normal_angle_visible_area.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    # Create a CSV writer
    csv_writer = csv.writer(csvfile)
    for i in range(len(t)):
            # To the csv table, write the frame number, normal vector, visible area, and timestamp 
            csv_writer.writerow([str(i),str(t_nv[t[i]]),str(t_va[t[i]]),t[i]])


# In[96]:


# Open the text file for writing the axis of rotation with the timestamps it is calculated between
axes = np.array(['X', 'Y', 'Z'])
txt_file_name = 'rotation_axis.txt'
with open(txt_file_name, "w") as file:
    for i in range(1, len(t)):
        max_val = max(abs(t_rotax[t[i]]))
        index = np.where(abs(t_rotax[t[i]])==max_val)
        file.write('Major axis of rotation is: '+str(axes[index][0]) + ', with rotation vector: ' + str( t_rotax[t[i]]) + ', between timestamps: '+str(t[i-1])+' and '+str(t[i])+'\n')


# In[ ]:





# In[ ]:





# In[ ]:




