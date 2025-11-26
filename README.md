# rotation-from-depth-map
Extracting Axis of rotation and the area of the largest rectangular face of a rotating cuboid from a depth map in a ROS bag file

# Approach
## Step 1 Create a depth image from the depth map
A depth map is created in int16 format. Using the int8 format resulted in a 2-channel depth map
where 1 is expected.
## Step 2 Create a 3D point cloud from a depth map
Assuming a focal length of 1, assuming the depth map is in SI units, and to conserve scale, and
the centre of the camera of the same as the centre of the image, which is an educated guess from
my previous experience. Values of depth greater than 5000m are clipped to ignore the
background in the distance.
## Step 3 Extract planes with RANSAC
Retrieve planes from each frame using RANSAC. With the parameters of 7000 minimum
samples and a residual threshold of 80.0, we get 2 planes from each frame.
## Step 4 Extract the normal vector and visible area of the plane from the 3D point cloud with
the greatest area
SVD is used to retrieve the normal vector. We use a randomly selected subset of 10000 for SVD,
as it is easier to calculate with this number of samples. For calculating the area, the two major
axes are selected so that the area of the convex hull can be calculated efficiently. The planes are
sorted by area, and the plane with the greatest area is selected for further processing. We store
the visible area and the normal vector in a dictionary indexed by the timestamp.
## Step 5 Calculate the angle between the camera normal and the normal vector of the selected
plane
We set the camera normal as -Z w.r.t. to the camera frame to get an angle consistent with the
illustration in the task description. This is done using the formula:

$θ= cos^{−1} \big( \frac{a. b}{ ∥a∥∥b∥ } \big)$
## Step 6 Calculate the axis of rotation between two contiguous frames
The axis of rotation is calculated using the cross product of the normal vector from contiguous
timestamps/frames.


# Data
The depth map of the scene is available in the files, depth.db3 and metadata.yaml.

# Code
Best to follow the steps in the jupyter notebook. It has been well commented and should be easy to peruse.

Any inputs on improving readability are always welcome.

