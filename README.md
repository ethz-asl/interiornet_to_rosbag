# interiornet_to_rosbag
Dataset tools for converting the InteriorNet dataset raw sequence data to a ROS bag.

Adapted from https://github.com/ethz-asl/scenenet_to_rosbag.

## The InteriorNet dataset
The InteriorNet dataset is split into the following:
1. 705 scenes with 3 ground truth trajectories and IMU data, color and depth images, ground truth instance images, with 1000 frames each, at 25 frames per second. Each trajectory is available in regular lighting and random lighting. *[HD1-HD6]*


2. 20000 scenes with 20 random views containing ground truth camera pose, color and depth images and ground truth instance and object class images. These views are also available in both regular lighting and random lighting. *[HD7]*


## How to use these tools
1. Clone this repository to the `src` folder of your catkin workspace, build your workspace and source it.

    ```bash
    cd <catkin_ws>/src
    git clone git@github.com:ethz-asl/interiornet_ros_tools.git
    catkin build
    source <catkin_ws>/devel/setup.bash
    ```

2. Download the entire InteriorNet dataset (very large) or only the trajectory files needed (request access at https://interiornet.org/), and unzip them in respective folders (HD[1-7]), eg. "interiorNet/data/HD7/3FO4IDEI1LAV_Dining_room". For HD1-HD6 trajectories, the corresponding ground truth zip file (it has the same name) has to be downloaded as well, and unzipped into the same folder, eg. "interiorNet/data/HD1/3FO4JXIK2PXE". 


3. Make the Python script executable and run it as a ROS node to convert data from an InteriorNet trajectory to a rosbag. The rosbag will contain a sequence of RGB and depth images, ground truth 2D instance label images, and relative transforms. Optionally, it can contain an nyu mask image, colored pointclouds of the scene, and colored pointclouds of ground truth instance segments.

    ```bash
    cd ../interiornet_to_rosbag && chmod +x nodes/interiornet_to_rosbag.py
    rosrun interiornet_ros_tools interiornet_to_rosbag.py --scene-path PATH/TO/SCENE --output-bag-path PATH/TO/BAG [--frame-step N] [--frame-limit N] [--light-type {original, random}] [--traj N] [--publish]

    ``` 
    Example for a trajectory from the HD1-HD6 scenes (note that --traj can take the value of 1, 3 or 7):
    ```bash
    rosrun interiornet_ros_tools interiornet_to_rosbag.py --scene-path ../data/interiorNet/data/HD1/3FO4JXIK2PXE --output-bag-path ../bags/HD7/3FO4JXIK2PXE.bag --light-type random --traj 3
    ```
    Example for a trajectory from the HD7 scenes (note that --traj N has no effect):
    ```bash
    rosrun interiornet_ros_tools interiornet_to_rosbag.py --scene-path ../data/interiorNet/data/HD7/3FO4IDI9FO3C_Guest_room --output-bag-path ../bags/HD7/3FO4IDI9FO3C_Guest_room.bag --frame-step 2 --frame-limit 10 --light-type original
    ```
    An alternative to writing the scene output to a bagfile is to publish it directly (note that this does not work in realtime for HD1-HD6 trajectories):
    ```bash
    rosrun interiornet_ros_tools interiornet_to_rosbag.py --scene-path ../data/interiorNet/data/HD7/3FO4IDI9FO3C_Guest_room --publish
    ```
    The output bag contains the following topics:
    ```bash
    # RGB and depth images
    /camera/rgb/camera_info         : sensor_msgs/CameraInfo
    /camera/rgb/image_raw           : sensor_msgs/Image
    /camera/depth/camera_info       : sensor_msgs/CameraInfo
    /camera/depth/image_raw         : sensor_msgs/Image        

    # Ground truth 2D instance segmentation image
    /camera/instances/image_raw     : sensor_msgs/Image
    
    # Colored NYU mask image
    /camera/instances/nyu_mask      : sensor_msgs/Image

    # Colored pointclouds of ground truth instance segments [Disabled by default]
    /interiornet_node/object_segment   : sensor_msgs/PointCloud2

    # Colored pointcloud of the scene                       [Disabled by default]
    /interiornet_node/scene            : sensor_msgs/PointCloud2

    # Transform from /scenenet_camera_frame to /world
    /tf                             : tf/tfMessage
    ```
    
    
## Additional data
The HD1-HD6 scenes also contain ground truth IMU, fisheye and panorama data. This is not implemented yet.
