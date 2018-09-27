#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import pcl

################################################################################
# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Helper function to convert from pcl to ros msg
def pcl_to_ros(pcl_array):
    """ Converts a pcl PointXYZRGB to a ROS PointCloud2 message

        Args:
            pcl_array (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud

        Returns:
            PointCloud2: A ROS point cloud
    """
    ros_msg = PointCloud2()

    ros_msg.header.stamp = rospy.Time.now()
    ros_msg.header.frame_id = "world"

    ros_msg.height = 1
    ros_msg.width = pcl_array.size

    ros_msg.fields.append(PointField(
                            name="x",
                            offset=0,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="y",
                            offset=4,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="z",
                            offset=8,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="rgb",
                            offset=16,
                            datatype=PointField.FLOAT32, count=1))

    ros_msg.is_bigendian = False
    ros_msg.point_step = 32
    ros_msg.row_step = ros_msg.point_step * ros_msg.width * ros_msg.height
    ros_msg.is_dense = False
    buffer = []

    for data in pcl_array:
        s = struct.pack('>f', data[3])
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value

        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        buffer.append(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))

    ros_msg.data = "".join(buffer)

    return ros_msg



# Helper function to covert from ros msg to pcl
def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message

        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
    """
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])

    pcl_data = pcl.PointCloud_PointXYZRGB()
    pcl_data.from_list(points_list)

    return pcl_data

# Helper cloud to convert from XYZRGB pcl to XYZ pcl
def XYZRGB_to_XYZ(XYZRGB_cloud):
    """ Converts a PCL XYZRGB point cloud to an XYZ point cloud (removes color info)

        Args:
            XYZRGB_cloud (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud

        Returns:
            PointCloud_PointXYZ: A PCL XYZ point cloud
    """
    XYZ_cloud = pcl.PointCloud()
    points_list = []

    for data in XYZRGB_cloud:
        points_list.append([data[0], data[1], data[2]])

    XYZ_cloud.from_list(points_list)
    return XYZ_cloud
################################################################################

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    # Parsing the ROS msg into PCL Format
    # For processing purposes we take a snapshot of the enviroment and work with
    # it for the whole pick list
    pcl_cloud = ros_to_pcl(pcl_msg)


    # TODO: Statistical Outlier Filtering
    # Removes outlier points whose position distance from neighbours exceeds
    # the specified threshold (removes noise)
    pcl_stat = pcl_cloud.make_statistical_outlier_filter()
    pcl_stat.set_mean_k(20)
    pcl_stat.set_std_dev_mul_thresh(0.001)
    pcl_stat_filtered = pcl_stat.filter()

    # TODO: Voxel Grid Downsampling
    # Downsampling the cloud for minimizing computation time
    pcl_vox = pcl_stat_filtered.make_voxel_grid_filter()
    leaf_size = 0.01
    pcl_vox.set_leaf_size(leaf_size,leaf_size,leaf_size)
    pcl_vox_filtered = pcl_vox.filter()

    # TODO: PassThrough Filter
    # Z axis filter to only show the desired reigon of the camera view
    pcl_pass = pcl_vox_filtered.make_passthrough_filter()
    axis_min = 0.6
    axis_max = 0.9
    axis_label = 'z'
    pcl_pass.set_filter_field_name(axis_label)
    pcl_pass.set_filter_limits(axis_min,axis_max)
    pcl_pass_filtered = pcl_pass.filter()

    # Y axis filter to remove the sides of the boxes, as they are considered
    # seperate clusters which would lead to false positives
    pcl_pass2 = pcl_pass_filtered.make_passthrough_filter()
    axis2_min = -0.5
    axis2_max = 0.5
    axis2_label = 'y'
    pcl_pass2.set_filter_limits(axis2_min,axis2_max)
    pcl_pass2.set_filter_field_name(axis2_label)
    pcl_pass2_filtered = pcl_pass2.filter()

    # TODO: RANSAC Plane Segmentation
    # Extracts the indices of the table (the object that satisfies the plane
    # model)
    pcl_seg = pcl_pass2_filtered.make_segmenter()
    pcl_seg.set_model_type(pcl.SACMODEL_PLANE)
    pcl_seg.set_method_type(pcl.SAC_RANSAC)
    max_dist = 0.01
    pcl_seg.set_distance_threshold(max_dist)
    inliers, coeffs = pcl_seg.segment()

    # TODO: Extract inliers and outliers

    # Extracts the points whose indeces are obtained from the segmenter (table)
    ext_inliers = pcl_pass2_filtered.extract(inliers,negative=False)
    # Extracts the points whose indices are not included in the segmenter
    # extraction (objects on table)
    ext_outliers = pcl_pass2_filtered.extract(inliers,negative=True)




    # TODO: Euclidean Clustering

    # Transforming the cloud to XYZ cloud format
    # Setting Euclidean Cluster parameters - distance threshold to be in a
    # specific cluster and cluster's min and max number of elements

    # Using kd-tree method to minimize computing time and resources
    pcl_white = XYZRGB_to_XYZ(ext_outliers)
    pcl_white_cluster = pcl_white.make_EuclideanClusterExtraction()
    dist_thresh = 0.03
    min_elements = 10
    max_elements = 10000
    pcl_white_cluster.set_MaxClusterSize(max_elements)
    pcl_white_cluster.set_MinClusterSize(min_elements)
    pcl_white_cluster.set_ClusterTolerance(dist_thresh)
    kd_tree = pcl_white.make_kdtree()
    pcl_white_cluster.set_SearchMethod(kd_tree)
    cluster_indices = pcl_white_cluster.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    # Generating random colors equal to the number of the found clusters and
    # setting a color to each one for visualization
    cluster_gen_colors = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for cluster_no,cluster_tuple in enumerate(cluster_indices):
        for indice in cluster_tuple:
            color_cluster_point_list.append([pcl_white[indice][0],
                                           pcl_white[indice][1],
                                           pcl_white[indice][2],
                                           rgb_to_float(cluster_gen_colors[cluster_no])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)


    # TODO: Convert PCL data to ROS messages
    # Converting the pcl to ROS msg format for publishing

    objects_ros = pcl_to_ros(ext_outliers)
    table_ros = pcl_to_ros(ext_inliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(objects_ros)
    pcl_table_pub.publish(table_ros)
    pcl_cluster_cloud.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []


    cloud_objects = ext_outliers

    models = [\
       'sticky_notes',
       'book',
       'snacks',
       'biscuits',
       'eraser',
       'soap2',
       'soap',
       'glue']

    model_name = "Test cluster"

    #for model_name in models:
    detected_pcl_labeled = []
    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        # Take the points from the original cluster with the cluster indices
        pcl_cluster = cloud_objects.extract(pts_list)

        label = []
        labeled_features = []
        sample_cloud = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector

        # Computing the color and normals histograms and concatenating them
        # together to obtain a feature vector
        chists = compute_color_histograms(sample_cloud, using_hsv=True)
        normals = get_normals(sample_cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        labeled_features.append([feature, model_name])


        # Make the prediction

        # Predicting each cluster using a previously trained SVM model
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(pcl_white[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster_cloud
        detected_objects.append(do)
        detected_pcl_labeled.append([label,pcl_cluster])


    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()

    # Appending all table points in table_list for future use in pr2_mover fun.
    table_list = []
    for point in ext_inliers:
        table_list.append([point[0],point[1],point[2],point[3]])

    pr2_mover(detected_pcl_labeled,table_list)



    try:
        pr2_mover(detected_pcl_labeled,table_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service

def pr2_mover(object_list,table_list):

    # TODO: Initialize variables
    pick_list = []
    found_cluster = []
    labels = []
    dict_list = []
    drop_point = []
    collision_list = []
    rem_list = []
    rem_cloud = pcl.PointCloud_PointXYZRGB()


    # TODO: Get/Read parameters
    # Read the required objects to pick up
    object_list_param = rospy.get_param('/object_list')


    # TODO: Parse parameters into individual variables
    for object_no in object_list_param:
        object_name = object_no['name']
        object_group = object_no['group']
        pick_list.append([object_name,object_group])


    # The node reciving the the collision cloud takes only the first msg and
    # builds on it without resetting which make us unable to make a different
    #collision cloud for each pickup due to objects already picked up

    # So we will make one collision cloud with all the objects that won't be
    # picked up
    # The code for generating different collision clouds is also included as
    # comments
    # Append all objects clusters in the collision map for now and later we
    # shall remove the clusters for the objects in the pick list
    for count in object_list:
        collision_list.append(count)

    # Remove clusters whose label exist in the pick list
    for order in pick_list:
        for no, collision in enumerate(collision_list):
            if order[0] == collision[0]:
                del collision_list[no]
    # A two nested FOR loops (objects list and pick list) with an if condition
    # in them would lead to append all of the clusters in the collision list
    # as the label of the object will be the same for one label in the pick list
    # and then when comparing it to the next label in the pick list they will be
    # different and hence the cluster will be added

    # Append all points of the table to rem_list which we will be using to make
    # the collision cloud
    for point in table_list:
        rem_list.append(point)

    # Append all the points of the objects which we will not be picking up
    for cluster in collision_list:
        for point in cluster[1]:
            rem_list.append(point)


    # TODO: Loop through the pick list
    for order in pick_list:
        print "REQUIRED OBJ : " , order[0]

        ######################################
        # used in generating different collision cloud for each pick up
        # object (not used)
        # >> Resetting the rem_cloud to the table points only for each pick up
        #for point in table_list:
        #    rem_list.append(point)
        ######################################


        # TODO: Get the PointCloud for a given object and obtain it's centroid
        # Loop through the objects labels to see if one is equal to the current
        # required object
        for no , count in enumerate(object_list):
            if order[0] == count[0]:
                found_cluster = count[1]
                labels.append(order[0])



                if order[1] == 'green':
                    arm = 'right'
                else:
                     arm = 'left'

                ######################################
                # used in generating different collision cloud for each pick up
                # object (not used)
                # >> Removing the objects labeled in the pick list so that it
                # doesn't satisfy the condition for the remaining labels in the
                # pick list and get appended
                #del collision_list[no]
                ######################################

        ######################################
        # used in generating different collision cloud for each pick up
        # object (not used)
        # >> Append all the remaining objects' points whose lable isn't equal to
        # the current required object or previous picked objects
        #for count in object_list:
        #    for point in count[1]:
        #        rem_list.append(point)
        ######################################


        # TODO: Create 'place_pose' for the object

        # Make a cloud from the rem_list and publish it as the collision cloud
        rem_cloud.from_list(rem_list)
        rem_cloud_ros = pcl_to_ros(rem_cloud)
        avoid_cloud.publish(rem_cloud_ros)
        rem_cloud_pub.publish(rem_cloud_ros)

        ######################################
        # used in generating different collision cloud for each pick up
        # object (not used)
        # >> Empty the rem_list for the use in next required object to pick up
        #del rem_list[:]
        ######################################


        # initializing the points' positions lists
        x_list=[]
        y_list=[]
        z_list=[]

        # For each required object add all the points' components to the lists
        for point in found_cluster:
            x_list.append(point[0])
            y_list.append(point[1])
            z_list.append(point[2])

        # Calculate the points mean value
        cent_x = np.sum(x_list)/len(x_list)
        cent_x = round(cent_x,3)
        cent_y = np.sum(y_list)/len(y_list)
        cent_y = round(cent_y,3)

        # Due to the camera prespective (looks downward onto objects), it
        # captures the points on top of the objects but not the bottom

        # So calculating the mean using the previous method leads to pulling
        # mean value up from the real center of the objects as more points exist
        # in the far end on the upper half than the lower half, and the robot
        # tries to grap in the space on top of the object

        # So we use the average between the max and min point instead and that
        # calculates the origin correctly and the robot is able to grab the
        # object
        cent_z = np.min(z_list) + ((np.max(z_list)-np.min(z_list))/2)
        cent_z = round(cent_z,3)


        centroids = []
        centroids.append([cent_x,cent_y,cent_z])

        # TODO: Assign the arm to be used for pick_place
        test_scene_num = Int32()
        test_scene_num.data = 2

        object_name = String()
        object_name.data = order[0]


        arm_name = String()
        arm_name.data = arm

        print "PICK POSITION"
        print cent_x
        print cent_y
        print cent_z

        # Set the point of the pick up for the ros msg
        pick_pose = Pose()
        pick_pose.position.x = cent_x
        pick_pose.position.y = cent_y
        pick_pose.position.z = cent_z

        # Check the dropbox file to check which box to drop in and the drop
        # point coordinates
        drop_param = rospy.get_param('/dropbox')
        for count in drop_param:
            if arm == count['name']:
                drop_point = count['position']

        # Set the point of the place for the ros msg
        place_pose = Pose()
        place_pose.position.x = drop_point[0]
        place_pose.position.y = drop_point[1]
        place_pose.position.z = drop_point[2]


        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)



        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    # TODO: Output your request parameters into output yaml file
    send_to_yaml('Objects_Positions',dict_list)



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('PR2_OR', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points",pc2.PointCloud2,pcl_callback,queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects",PointCloud2,queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table",PointCloud2,queue_size=1)
    pcl_cluster_cloud = rospy.Publisher("/pcl_cluster",PointCloud2,queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers",Marker,queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects",DetectedObjectsArray,queue_size=1)
    avoid_cloud = rospy.Publisher("/pr2/3d_map/points",PointCloud2,queue_size=3)
    rem_cloud_pub = rospy.Publisher("/rem_cloud",PointCloud2,queue_size=1)


    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
