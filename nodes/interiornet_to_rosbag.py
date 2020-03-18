#!/usr/bin/env python2

import os
import sys
import argparse
import numpy as np
import rosbag
import rospy
import cv2
import pandas as pd

from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, TransformStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf.msg import tfMessage
import sensor_msgs.point_cloud2 as pc2
import tf


def normalize(v):
    return v / np.linalg.norm(v)


def world_to_camera_with_pose(view_pose):
    camera_pose = view_pose[:3]
    lookat_pose = view_pose[3:6]
    up = view_pose[6:]
    R = np.diag(np.ones(4))
    R[2, :3] = normalize(lookat_pose - camera_pose)
    R[0, :3] = normalize(np.cross(R[2, :3], (up - camera_pose)))
    R[1, :3] = -normalize(np.cross(R[0, :3], R[2, :3]))
    T = np.diag(np.ones(4))
    T[:3, 3] = -camera_pose
    return R.dot(T)


def camera_to_world_with_pose(view_pose):
    return np.linalg.inv(world_to_camera_with_pose(view_pose))


def get_camera_info():
    # Not entirely sure if it is correct lol. Check in ground truth ... cam0.ccam

    camera_info = CameraInfo()
    camera_info.height = 480
    camera_info.width = 640

    camera_info.distortion_model = "plumb_bob"
    camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    camera_info.R = np.ndarray.flatten(np.identity(3))
    camera_info.K = np.ndarray.flatten(np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]))
    camera_info.P = np.ndarray.flatten(np.array([[600., 0., 320., 0.], [0., 600., 240., 0.], [0., 0., 1., 0.]]))
    return camera_info


def euclidean_ray_length_to_z_coordinate(depth_image, camera_model):
    center_x = camera_model.cx()
    center_y = camera_model.cy()

    constant_x = 1 / camera_model.fx()
    constant_y = 1 / camera_model.fy()

    vs = np.array(
        [(v - center_x) * constant_x for v in range(0, depth_image.shape[1])])
    us = np.array(
        [(u - center_y) * constant_y for u in range(0, depth_image.shape[0])])

    return (np.sqrt(
        np.square(depth_image / 1000.0) /
        (1 + np.square(vs[np.newaxis, :]) + np.square(us[:, np.newaxis]))) *
            1000.0).astype(np.uint16)


def pack_bgr(blue, green, red):
    # Pack the 3 BGR channels into a single UINT32 field as RGB.
    return np.bitwise_or(
        np.bitwise_or(
            np.left_shift(red.astype(np.uint32), 16),
            np.left_shift(green.astype(np.uint32), 8)), blue.astype(np.uint32))


def convert_bgrd_to_pcl(bgr_image, depth_image, camera_model):
    center_x = camera_model.cx()
    center_y = camera_model.cy()

    constant_x = 1 / camera_model.fx()
    constant_y = 1 / camera_model.fy()

    pointcloud_xzyrgb_fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
    ]

    vs = np.array(
        [(v - center_x) * constant_x for v in range(0, depth_image.shape[1])])
    us = np.array(
        [(u - center_y) * constant_y for u in range(0, depth_image.shape[0])])

    # Convert depth from mm to m.
    depth_image = depth_image / 1000.0

    x = np.multiply(depth_image, vs)
    y = depth_image * us[:, np.newaxis]

    stacked = np.ma.dstack((x, y, depth_image, bgr_image))
    compressed = stacked.compressed()
    pointcloud = compressed.reshape((int(compressed.shape[0] / 6), 6))

    pointcloud = np.hstack((pointcloud[:, 0:3],
                            pack_bgr(*pointcloud.T[3:6])[:, None]))
    pointcloud = [[point[0], point[1], point[2], point[3]]
                  for point in pointcloud]

    pointcloud = pc2.create_cloud(Header(), pointcloud_xzyrgb_fields,
                                  pointcloud)
    return pointcloud


def write_msg(topic, msg, output_bag, publishers, publish, timestamp=None):
    if publish:
        publishers[topic].publish(msg)
    else:
        output_bag.write(topic, msg, timestamp)


def write_transform(view_pose, timestamp, frame_id, output_bag, publishers, publish):
    scale, shear, angles, transl, persp = tf.transformations.decompose_matrix(
        camera_to_world_with_pose(view_pose))
    rotation = tf.transformations.quaternion_from_euler(*angles)

    trans = TransformStamped()
    trans.header.stamp = timestamp
    trans.header.frame_id = 'world'
    trans.child_frame_id = frame_id
    trans.transform.translation.x = transl[0]
    trans.transform.translation.y = transl[1]
    trans.transform.translation.z = transl[2]
    trans.transform.rotation.x = rotation[0]
    trans.transform.rotation.y = rotation[1]
    trans.transform.rotation.z = rotation[2]
    trans.transform.rotation.w = rotation[3]

    msg = tfMessage()
    msg.transforms.append(trans)

    #output_bag.write('/tf', msg, timestamp)
    write_msg("/tf", msg, output_bag, publishers, publish, timestamp)


def parse_frames(scene_path, scene_type, traj):
    if scene_type == 7:
        path = os.path.join(scene_path, "cam0.render")
    else:
        path = os.path.join(scene_path, "velocity_angular_{}_{}/cam0.render".format(traj, traj))

    # Read cam0.render file for camera pose.
    try:
        # Skip first 3 rows and every second row.
        lines = pd.read_csv(path, sep=" ", header=None, skiprows=3).iloc[::2, :]
    except IOError:
        print('cam0.render not found at location: {0}'.format(scene_path))
        print(path)
        sys.exit('Please ensure you have unzipped the files to the data directory.')

    # First three lines are comments.
    # Two lines per frame (shutter speed is 0).
    # Poses are written in the file as eye, look-at, up (see cam0.render)
    num_frames = lines.shape[0] / 2
    data = lines.to_numpy()

    view_poses = data[:, 1:]
    times = data[:, 0]

    # Prevent time being zero for HD7 scenes.
    if scene_type == 7:
        times = times + 1.
    elif scene_type < 7:
        times = times / 1e9

    return times, view_poses


def convert(scene_path, scene_type, light_type, traj, frame_step, to_frame, output_bag, publishers, publish):
    frame_id = "/interiornet_camera_frame"

    # Write RGB and depth images.
    write_rgbd = True
    # Write instance image.
    write_instances = True
    # Write colorized instance image.
    write_instances_nyu_mask = False
    # Write colored pointclouds of the instance segments.
    write_object_segments = False
    # Write colored pointclouds of the whole scene.
    write_scene_pcl = True

    # Set camera information and model.
    camera_info = get_camera_info()
    camera_model = PinholeCameraModel()
    camera_model.fromCameraInfo(camera_info)

    # Initialize some vars.
    header = Header(frame_id=frame_id)
    cvbridge = CvBridge()

    # Read frame camera time and view_pose list.
    times, view_poses = parse_frames(scene_path, scene_type, traj)

    # Start writing scene to rosbag.
    print('Writing scene from location: ' + format(scene_path))
    '''
    We only have camera pose here.
    '''
    view_idx = 0
    while (not rospy.is_shutdown() and view_idx < to_frame
           and view_idx < np.shape(view_poses)[0]):
        view_pose = view_poses[view_idx, :]
        if publish:
            timestamp = rospy.Time.now()
        else:
            timestamp = rospy.Time.from_sec(times[view_idx])
        write_transform(view_pose, timestamp, frame_id, output_bag, publishers, publish)
        header.stamp = timestamp

        if light_type != "original" and light_type != "random":
            print("ERROR: light type not available. ")
            sys.exit("Please use light_type 'original' or 'random'")

        img_id = view_idx

        # Read RGB, Depth and Instance images for the current view.
        if scene_type == 7 and light_type == "original":
            photo_path = os.path.join(scene_path, "cam0/data/{}.png".format(img_id))
        elif scene_type == 7 and light_type == "random":
            photo_path = os.path.join(scene_path, "random_lighting_cam0/data/{}.png".format(img_id))
        elif scene_type < 7:
            photo_path = os.path.join(scene_path, "{}_{}_{}".format(light_type, traj, traj),
                                      "cam0/data/{}.png".format(img_id))

        if scene_type == 7:
            depth_path = os.path.join(scene_path, "depth0/data/{}.png".format(img_id))
        elif scene_type < 7:
            depth_path = os.path.join(scene_path, "{}_{}_{}".format(light_type, traj, traj),
                                      "depth0/data/{}.png".format(img_id))

        if scene_type == 7:
            instance_path = os.path.join(scene_path, "label0/data/{}_instance.png".format(img_id))
        elif scene_type < 7:
            instance_path = os.path.join(scene_path, "{}_{}_{}".format(light_type, traj, traj),
                                         "label0/data/{}_instance.png".format(img_id))

        if scene_type == 7:
            nyu_mask_path = os.path.join(scene_path, "label0/data/{}_nyu_mask.png".format(img_id))
        elif scene_type < 7:
            nyu_mask_path = os.path.join(scene_path, "{}_{}_{}".format(light_type, traj, traj),
                                         "label0/data/{}_nyu_mask.png".format(img_id))

        if not os.path.exists(photo_path):
            print("InteriorNet RGB-D data not found at {0}".format(photo_path))
            sys.exit("Please ensure you have downloaded the data.")

        bgr_image = cv2.imread(photo_path, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        instance_image = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)

        # Transform depth values from the Euclidean ray length to the z coordinate.
        depth_image = euclidean_ray_length_to_z_coordinate(
            depth_image, camera_model)

        if write_object_segments:
            # Write all the instances in the current view as pointclouds.
            instances_in_current_frame = np.unique(instance_image)

            for instance in instances_in_current_frame:
                instance_mask = np.ma.masked_not_equal(instance_image,
                                                       instance).mask
                masked_depth_image = np.ma.masked_where(
                    instance_mask, depth_image)

                # Workaround for when 2D mask is only False values and collapses to a single boolean False.
                if not instance_mask.any():
                    instance_mask_3D = np.broadcast_arrays(
                        instance_mask[np.newaxis, np.newaxis, np.newaxis],
                        bgr_image)
                else:
                    instance_mask_3D = np.broadcast_arrays(
                        instance_mask[:, :, np.newaxis], bgr_image)

                masked_bgr_image = np.ma.masked_where(instance_mask_3D[0],
                                                      bgr_image)

                object_segment_pcl = convert_bgrd_to_pcl(
                    masked_bgr_image, masked_depth_image, camera_model)
                object_segment_pcl.header = header
                #output_bag.write('/scenenet_node/object_segment',
                #                 object_segment_pcl, timestamp)
                write_msg('/interiornet_node/object_segment', object_segment_pcl, output_bag, publishers, publish)

        if write_scene_pcl:
            # Write the scene for the current view as pointcloud.
            scene_pcl = convert_bgrd_to_pcl(bgr_image, depth_image,
                                            camera_model)
            scene_pcl.header = header
            #output_bag.write('/scenenet_node/scene', scene_pcl, timestamp)
            write_msg('/interiornet_node/scene', scene_pcl, output_bag, publishers, publish)

        if write_rgbd:
            # Write the RGBD data.
            bgr_msg = cvbridge.cv2_to_imgmsg(bgr_image, "bgr8")
            bgr_msg.header = header
            #output_bag.write('/camera/rgb/image_raw', bgr_msg, timestamp)
            write_msg('/camera/rgb/image_raw', bgr_msg, output_bag, publishers, publish)

            depth_msg = cvbridge.cv2_to_imgmsg(depth_image, "16UC1")
            depth_msg.header = header
            #output_bag.write('/camera/depth/image_raw', depth_msg, timestamp)
            write_msg('/camera/depth/image_raw', depth_msg, output_bag, publishers, publish)

            camera_info.header = header

            #output_bag.write('/camera/rgb/camera_info', camera_info, timestamp)
            #output_bag.write('/camera/depth/camera_info', camera_info,
            #                 timestamp)
            write_msg('/camera/rgb/camera_info', camera_info, output_bag, publishers, publish)
            write_msg('/camera/depth/camera_info', camera_info, output_bag, publishers, publish)

        if write_instances:
            # Write the instance data.
            instance_msg = cvbridge.cv2_to_imgmsg(instance_image, "16UC1")
            instance_msg.header = header

            #output_bag.write('/camera/instances/image_raw', instance_msg,
            #                 timestamp)
            write_msg('/camera/instances/image_raw', instance_msg, output_bag, publishers, publish)

        if write_instances_nyu_mask:
            instance_nyu_mask_image = cv2.imread(nyu_mask_path, cv2.IMREAD_UNCHANGED)

            # Write the instance data colorized.
            instance_nyu_mask_msg = cvbridge.cv2_to_imgmsg(instance_nyu_mask_image,
                                                      "16UC3")
            instance_nyu_mask_msg.header = header

            #output_bag.write('/camera/instances/nyu_mask', instance_nyu_mask_msg,
            #                 timestamp)
            write_msg('/camera/instances/nyu_mask', instance_nyu_mask_msg, output_bag, publishers, publish)

        print("Dataset timestamp: " + '{:4}'.format(timestamp.secs) + "." +
              '{:09}'.format(timestamp.nsecs) + "     Frame: " +
              '{:3}'.format(view_idx + 1) + " / " + str(np.shape(view_poses)[1]))

        view_idx += frame_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='''%(prog)s [-h] --scene-path PATH [--output-bag-path PATH] [--frame-step NUM] \
            [--frame-limit NUM] [--light-type {original, random}] [--traj NUM] [--publish]''',
        description='Convert data from a InteriorNet RGB-D trajectory to a rosbag.'
    )
    parser.add_argument(
        "--scene-path",
        required=True,
        help="path to the scene folder, e.g. /home/[user]/catkin_ws/data/interiorNet/data/HD7/3FO4II2X5NUD_Guest_room",
        metavar="PATH")
    parser.add_argument(
        "--output-bag-path",
        default="interiornet_{}.bag",
        help="output path the rosbag will be written to.",
        metavar="PATH")
    parser.add_argument(
        "--frame-step",
        default=1,
        type=int,
        help="write every NUM frames to bag (Bigger than 1 means NUM - 1 frames will be skipped for each step).",
        metavar="NUM")
    parser.add_argument(
        "--frame-limit",
        default=np.inf,
        type=int,
        help="only write NUM frames to the bag (Default: infinite)",
        metavar="NUM")
    parser.add_argument(
        "--light-type",
        default="original",
        help="Choose the desired lighting options.",
        choices=['original', 'random'])
    parser.add_argument(
        "--traj",
        default=1,
        type=int,
        help="Choose the desired trajectory.",
        choices=[1, 3, 7],
        metavar="NUM")
    parser.add_argument(
        "--publish",
        dest='publish',
        action='store_true',
        help="If set, publishes the data directly and does not save as bag file.")
    parser.set_defaults(publish=False)

    # TODO: This is only for HD7 scenes. HD1-6 scenes have different format
    #       and need to be implemented in the future.

    args = parser.parse_args()
    scene_path = args.scene_path
    frame_step = args.frame_step
    output_bag_path = args.output_bag_path
    to_frame = args.frame_limit
    light_type = args.light_type
    traj = args.traj
    publish = args.publish
    print(publish)

    # Read all scene paths from the scene list file.
    # with open(scene_list_path) as f:
    #     file_paths = f.readlines()
    # scene_paths = [x.strip('.zip\n') for x in file_paths]

    publishers = {}
    if publish:
        publishers = {
            '/tf': rospy.Publisher('/tf', tfMessage, queue_size=5),
            '/interiornet_node/object_segment': rospy.Publisher('/interiornet_node/object_segment', PointCloud2, queue_size=5),
            '/interiornet_node/scene': rospy.Publisher('/interiornet_node/scene', PointCloud2, queue_size=5),
            '/camera/rgb/image_raw': rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=5),
            '/camera/depth/image_raw': rospy.Publisher('/camera/depth/image_raw', Image, queue_size=5),
            '/camera/rgb/camera_info': rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=5),
            '/camera/depth/camera_info': rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=5),
            '/camera/instances/image_raw': rospy.Publisher('/camera/instances/image_raw', Image, queue_size=5),
            '/camera/instances/nyu_mask': rospy.Publisher('/camera/instances/nyu_mask', Image, queue_size=5),
        }

    rospy.init_node('interiornet_node', anonymous=True)

    scene_path_split = scene_path.rsplit('/')
    scene_type = int(scene_path_split[-2][2])
    scene_name = scene_path_split[-1]

    if publish:
        print("Waiting for subscribers to connect...")
        found = False
        while not found:
            for key, pub in publishers.items():
                if pub.get_num_connections() > 0:
                    found = True
                    break

        # Wait for the subscribers to connect properly. Better would to wait
        # for all relevant subscribers to connect.
        print("Subscriber found. Waiting 1 seconds before publishing.")
        rospy.sleep(1.)

    if not output_bag_path.endswith(".bag"):
        output_bag_path = output_bag_path + ".bag"
    output_bag = rosbag.Bag(output_bag_path, 'w')
    try:
        convert(scene_path, scene_type, light_type, traj, frame_step, to_frame, output_bag, publishers, publish)
    except rospy.ROSInterruptException:
        pass
    finally:
        output_bag.close()

