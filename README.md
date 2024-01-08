Leg Tracker
===========
![ROS](https://img.shields.io/badge/ROS-Noetic-brightgreen.svg)  ![OS](https://img.shields.io/badge/OS-Ubuntu%2020.04-orange.svg ) ![OpenCV](https://img.shields.io/badge/OpenCV-4.2-blue.svg)


Usage
-------------------
- `cd ~/catkin_ws/src`
- `git clone git@github.com:angusleigh/leg_tracker.git`
- install SciPy:
    - `sudo apt install python-scipy`
- install pykalman (http://pykalman.github.io/#installation or https://pypi.python.org/pypi/pykalman):
    - `sudo pip install pykalman`
- `cd ~/catkin_ws`    
- `catkin_make`
- `roslaunch leg_tracker joint_leg_tracker.launch`

Use Docker
------------------
To build this project use following command

```bash
make docker_build
```
To run this project use the following command
```bash
make docker_run
```

Demos
-------------------
- try `roslaunch leg_tracker demo_stationary_simple_environment.launch` to see a demo of it working on a stationary robot.
- there are a bunch more demo launch files in the /launch directory.
- video: [youtu.be/6k8y72AQG-Y](https://youtu.be/6k8y72AQG-Y), which also includes a brief explaination of the method.


Subscribed topics
-------------------
- /scan
    - you can change the subscribed scan topic by changing the parameter "scan_topic" in the launch file.
- /tf
    - to get the transforms. You should specify the fixed frame by the parameter "fixed_frame" in the launch file.


Published topics
-------------------
- /people_tracked
    - the positions and velocities of the tracked people as a PersonArray message (see: leg_tracker/msg/PersonArray.msg for the message specification).
- /visualization_marker
    - markers for the tracked people, other tracked objects and the centroids of the detected clusters.


Important parameters
-------------------
- scan_topic
- scan_frequency
    - right now, it's only been tuned for 7.5, 10 or 15Hz laser scanners. If you're using something faster than this, I would recommend downsampling to one of these frequencies.
- fixed_frame
- publish_people_frame
    - frame_id for /people_tracked messages. Default: fixed_frame.

Other parameters
-------------------
- max_detected_clusters
    - If the program is running slow because there are a lot of objects to be tracked, you can set this so it only tracks the "max_detected_clusters" closest objects. Default: -1 (i.e., unlimited).
- forest_file
    - where to find the configuration file for the OpenCV random forest which has been trained to determine the probability of a scan cluster being a human leg based on its shape.
- detection_threshold
    - Confidence threshold in leg shape to publish detect clusters. We found be results publishing and tracking all clusters, regardless of confidence in leg shape. Default: -1 (i.e., no threshold).
- cluster_dist_euclid
    - Euclidian distance for clustering of laser scan points. Any two laser scan points which are closer than this distance will be clustered together. If you change this, you might want to re-train the leg detector. Default: 0.13 (m).
- min_points_per_cluster
    - Minimum scan points for a cluster to be valid. If a cluster has less than this, then we assume it's noise and delete it. Default: 3.
- max_detect_distance
    - Maximum distance from laser scanner to consider detected clusters. Farther is better but decreases run speed. Default: 10.0 (m).
- max_leg_pairing_dist
    - Maximum separation of legs. Legs further apart than this distance will not be matched. Default: 0.8 (m).
- confidence_threshold_to_maintain_track
    - How high the confidence needs to be of the current track's leg shape to initiate and maintain a person track. Higher values will yield lower false positive people tracks and lower values higher false negatives. Will likely need re-tuning if you retrain the leg detector. Default: 0.1.
- publish_occluded
    - If people who weren't seen in the previous scan should be published to the /people_tracked topic. Default: True.
- use_scan_header_stamp_for_tfs
    - If you should use the scan headers to do tfs or just use the latest transform available. If you're trying to do repeatable tests on recorded bags than it might be useful to use the scan headers. Default: False (i.e., use latest tf available).
- display_detected_people
    - Previously used only for debugging.  Default: False.
- dist_travelled_together_to_initiate_leg_pair
    - How far a candidate leg pair must travel to initiate a person track. Default: 0.5 (m).
- in_free_space_threshold
    - A threshold to determine whether a track should be considered in free space or occupied space. Default: 0.06.
- confidence_percentile
    - Confidence percentile for matching of clusters to tracks. Larger confidence percentile generates larger gates for the tracks to match to new clusters. You should be able to view the size of the gates in Rviz as circular visualization markers. Default: 0.9.
- max_std
    - Maximum standard deviation of covariance of a track before it is deleted: Default: 0.9.


**Compatible laser scanners**

- right now it's trained for laser scanners of 0.33 degree resolution. Different resolutions should still be runnable but would probably require retraining (see below) to get optimal performance.


Retraining the leg detector
-------------------
- gather positive examples by setting the scanner up in a populated area where you specify a bounding box (or alternatively, an arc), within which only peoples' legs will appear.
- record scan data in a rosbag.
- run extract_positive_training_clusters.launch on the recorded data to have it automatically annotate where the legs are in the scans. It will put a marker on any clusters that fall within the specified bounding box. You can check if the right clusters have been annotated by playing back the generated rosbag with the "--pause" option and stepping through it and visually checking in Rviz where the markers lie in the scan.
- gather negative examples by moving the scanner on a moble platform around empty rooms.
- run train_leg_detector and include the positive and negative examples as command line arguments (see the train_leg_detector.launch for an example).
- a .yaml file will be output with the name of the "save_file" parameter in the train_leg_detector launch file.
- set the launch file for the joint_leg_tracker to load the .yaml file you produced.
- when running the joint_leg_tracker with your new detector, you will likely need to retune the "confidence_threshold_to_maintain_track" parameter so you get the desired tradeoff between false positive/false negative person tracks.


Under the hood
-------------------
**Files for running**

- src/laser_processor.cpp - scan processing stuff. E.g., splitting scan into clusters based on euclidian distance.
- src/cluster_features.cpp - calculates geometric features of scan clusters. Used by detect_leg_clusters.
- src/detect_leg_clusters.cpp - detects leg clusters in the scan and publishes them to /detected_leg_clusters.
- scripts/joint_leg_tracker.py - uses detect_leg_clusters and a global nearest neighbour tracking method to track people. They're published as markers to rviz and to a people_tracked topic.
- scripts/individual_leg_tracker.py - alternative to joint_leg_tracker. I found better performance with the joint_leg_tracker.


** Files for retraining the detector **

- src/extract_positive_training_clusters.cpp - extracts positive example clusters from scans in a rosbag. They're saved to another rosbag to be used for training.
- src/train_leg_detector.cpp - train the random forest classifier using the positive example clusters and negative scans examples.


Acknowledgement
-------------------
We used some code from the leg_detector package: http://wiki.ros.org/leg_detector.
It does fundamentally the same task, but we found this version was more capable of tracking people longer, not mismatching people and producing less false positives in cases where a grid occupancy map is not provided a priori. See the reference paper and the leg_tracker_benchmarks repo for more details.


License
-------------------
The entire repo is released under a BSD license. The source files forked from the leg_detector package maintain their BSD licenses (2008, Willow Garage, Inc.) in their respective files. All other files are covered by the LICENSE in the root of this repo.


TODO
-------------------
- port tracker to C++ to improve speed
- use BFL's kalman filter in place of pykalman so users aren't required to install the external dependancy
- integrate a priori occupancy grid maps or SLAM gmapping maps to speed up detection and reduce false positives


Reference
-------------------
A. Leigh, J. Pineau, N. Olmedo and H. Zhang, Person Tracking and Following with 2D Laser Scanners, International Conference on Robotics and Automation (ICRA), Seattle, Washington, USA, 2015. [pdf](https://www.cs.mcgill.ca/~aleigh1/ICRA_2015.pdf)

Citations appreciated!
