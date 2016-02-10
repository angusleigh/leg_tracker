#!/usr/bin/python

import rospy
from leg_tracker.msg import Person, PersonArray, Leg, LegArray
from visualization_msgs.msg import Marker
from pykalman import KalmanFilter # Third party library. To install: http://pykalman.github.io/#installation
import numpy as np
from munkres import Munkres # Third party library. For the minimum matching assignment problem. To install: https://pypi.python.org/pypi/munkres 
import random
from collections import deque
import math
import scipy.stats
import scipy.spatial
from geometry_msgs.msg import PointStamped
import tf
import copy
import timeit
import sys


class DetectedCluster:
    """
    A detected scan cluster. Not yet associated to an existing track.
    """
    def __init__(self, pos_x, pos_y, confidence):
        """
        Constructor
        """        
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.confidence = confidence


class TrackedPerson:
    """
    A tracked person 
    """
    new_person_id_num = 1

    def __init__(self, leg_1, leg_2):
        """
        Constructor
        """           
        self.leg_1 = leg_1
        self.leg_2 = leg_2       
        self.id_num = TrackedPerson.new_person_id_num
        self.colour = (random.random(), random.random(), random.random())
        TrackedPerson.new_person_id_num += 1


class PotentialLegPair:
    """ 
    A potential (i.e., not yet validated) person/pair of legs
    If validated, we think they represent a person
    """    
    def __init__(self, leg_1, leg_2):
        """
        Constructor
        """         
        self.leg_1 = leg_1
        self.leg_2 = leg_2
        self.leg_1_initial_dist_travelled = leg_1.dist_travelled
        self.leg_2_initial_dist_travelled = leg_2.dist_travelled
        self.validated_person = False


class ObjectTracked:
    """
    A tracked object. Could be a person leg, entire person or any arbitrary object in the laser scan.
    """
    new_leg_id_num = 1

    def __init__(self, x, y, now, confidence): 
        """
        Constructor
        """              
        self.id_num = ObjectTracked.new_leg_id_num
        ObjectTracked.new_leg_id_num += 1
        self.colour = (random.random(), random.random(), random.random())
        self.last_seen = now
        self.seen_in_current_scan = True
        self.times_seen = 1
        self.confidence = confidence
        self.dist_travelled = 0.
        self.person = None
        self.deleted = False

        # People are tracked via a constant-velocity Kalman filter with a Gaussian acceleration distrubtion
        # Kalman filter params were found by hand-tuning. 
        # A better method would be to use data-driven EM find the params. 
        # The important part is that the observations are "weighted" higher than the motion model 
        # because they're more trustworthy and the motion model kinda sucks
        scan_frequency = rospy.get_param("scan_frequency", 7.5)
        delta_t = 1./scan_frequency
        if scan_frequency > 7 and scan_frequency < 8:
            std_process_noise = 0.06666
        elif scan_frequency > 9 and scan_frequency < 11:
            std_process_noise = 0.05
        elif scan_frequency > 14 and scan_frequency < 16:
            std_process_noise = 0.03333
        else:
            print "Scan frequency needs to be either 7.5, 10 or 15 or the standard deviation of the process noise needs to be tuned to your scanner frequency"
        std_pos = std_process_noise
        std_vel = std_process_noise
        std_obs = 0.1
        var_pos = std_pos**2
        var_vel = std_vel**2
        var_obs_local = std_obs**2
        self.var_obs = (std_obs + 0.4)**2

        self.filtered_state_means = np.array([x, y, 0, 0])
        self.pos_x = x
        self.pos_y = y
        self.vel_x = 0
        self.vel_y = 0

        self.filtered_state_covariances = 0.5*np.eye(4) 

        # Constant velocity motion model
        transition_matrix = np.array([[1, 0, delta_t,        0],
                                      [0, 1,       0,  delta_t],
                                      [0, 0,       1,        0],
                                      [0, 0,       0,        1]])

        # Oberservation model. Can observe pos_x and pos_y (unless person is occluded - we deal with this later). 
        observation_matrix = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0]])

        transition_covariance = np.array([[var_pos,       0,       0,       0],
                                          [      0, var_pos,       0,       0],
                                          [      0,       0, var_vel,       0],
                                          [      0,       0,       0, var_vel]])

        observation_covariance =  var_obs_local*np.eye(2)

        self.kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )


    def update(self, observations):
        """
        Update our tracked object with new observations
        """        
        self.filtered_state_means, self.filtered_state_covariances = (
            self.kf.filter_update(
                self.filtered_state_means,
                self.filtered_state_covariances,
                observations
            )
        )

        delta_dist_travelled = ((self.pos_x - self.filtered_state_means[0])**2 + (self.pos_y - self.filtered_state_means[1])**2)**(1./2.) 
        if delta_dist_travelled > 0.01:
            self.dist_travelled += delta_dist_travelled

        self.pos_x = self.filtered_state_means[0]
        self.pos_y = self.filtered_state_means[1]
        self.vel_x = self.filtered_state_means[2]
        self.vel_y = self.filtered_state_means[3]
    


class KalmanMultiTracker:    
    """
    Tracker for tracking all the people and objects
    """
    max_cost = 9999999
    def __init__(self):
        """
        Constructor
        """      
        self.objects_tracked = []
        self.potential_leg_pairs = set()
        self.potential_leg_pair_initial_dist_travelled = {}
        self.people_tracked = []
        self.prev_track_marker_id = 0
        self.prev_person_marker_id = 0
        self.listener = tf.TransformListener()
        random.seed(1) 

        # Get ROS params
        self.fixed_frame = rospy.get_param("fixed_frame", "odom")
        self.max_leg_pairing_dist = rospy.get_param("max_leg_pairing_dist", 0.8)
        self.confidence_threshold_to_maintain_track = rospy.get_param("confidence_threshold_to_maintain_track", 0.1)
        self.publish_occluded = rospy.get_param("publish_occluded", True)
        self.publish_people_frame = rospy.get_param("publish_people_frame", self.fixed_frame)
        self.use_scan_header_stamp_for_tfs = rospy.get_param("use_scan_header_stamp_for_tfs", False)
        self.publish_detected_people = rospy.get_param("display_detected_people", False)        
        self.dist_travelled_together_to_initiate_leg_pair = rospy.get_param("dist_travelled_together_to_initiate_leg_pair", 0.5)
        self.scan_frequency = rospy.get_param("scan_frequency", 7.5)
        self.confidence_percentile = rospy.get_param("confidence_percentile", 0.90)
        self.max_std = rospy.get_param("max_std", 0.9)

        self.mahalanobis_dist_gate = scipy.stats.norm.ppf(1.0 - (1.0-self.confidence_percentile)/2., 0, 1.0)
        self.max_cov = self.max_std**2
        self.latest_scan_header_stamp_with_tf_available = rospy.get_rostime()

    	# ROS publishers
        self.people_tracked_pub = rospy.Publisher('people_tracked', PersonArray, queue_size=300)
        self.people_detected_pub = rospy.Publisher('people_detected', PersonArray, queue_size=300)
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=300)

        # ROS subscribers         
        self.detected_clusters_sub = rospy.Subscriber('detected_leg_clusters', LegArray, self.detected_clusters_callback)      

        rospy.spin() # So the node doesn't immediately shut down
                    
        
    def match_detections_to_tracks_global_nearest_neighbour(self, objects_tracked, objects_detected):
        """
        Match detected objects to existing object tracks using a global nearest neighbour data association
        """        
        matched_tracks = {}

        # Populate match_dist matrix of mahalanobis_dist between every detection and every track
        match_dist = [] # matrix of probability of matching between all people and all detections.   
        eligable_detections = [] # Only include detections in match_dist matrix if they're in range of at least one track to speed up munkres
        for detect in objects_detected: 
            at_least_one_track_in_range = False
            new_row = []
            for track in objects_tracked:
                # Use mahalanobis dist to do matching
                cov = track.filtered_state_covariances[0][0] + track.var_obs # cov_xx == cov_yy == cov
                mahalanobis_dist = math.sqrt(((detect.pos_x-track.pos_x)**2 + (detect.pos_y-track.pos_y)**2)/cov) # ref: http://en.wikipedia.org/wiki/Mahalanobis_distance#Definition_and_properties
                if mahalanobis_dist < self.mahalanobis_dist_gate:
                    new_row.append(mahalanobis_dist)
                    at_least_one_track_in_range = True
                else:
                    new_row.append(self.max_cost)
            # If the detection is within range of at least one person track, add it as an eligable detection in the munkres matching 
            if at_least_one_track_in_range: 
                match_dist.append(new_row)
                eligable_detections.append(detect)

        # Run munkres on match_dist to get the lowest cost assignment
        if match_dist:
            munkres = Munkres()
            # self.pad_matrix(match_dist, pad_value=self.max_cost) # I found no difference when padding it 
            indexes = munkres.compute(match_dist)
            for elig_detect_index, track_index in indexes:
                if match_dist[elig_detect_index][track_index] < self.mahalanobis_dist_gate:
                    detect = eligable_detections[elig_detect_index]
                    track = objects_tracked[track_index]
                    matched_tracks[track] = detect

        return matched_tracks

      
    def detected_clusters_callback(self, detected_clusters_msg):  
        """
        Callback for every time detect_leg_clusters publishes new sets of detected clusters. 
        It will try to match the newly detected clusters with tracked clusters from previous frames.
        """  
        now = detected_clusters_msg.header.stamp
       
        detected_clusters = []
        for cluster in detected_clusters_msg.legs:
            detected_clusters.append(DetectedCluster(cluster.position.x, cluster.position.y, cluster.confidence))
        
		 # Propogate existing tracks
        propogated = copy.deepcopy(self.objects_tracked)
        for track in propogated:
            track.update(np.ma.masked_array(np.array([0, 0]), mask=[1,1])) # Update copied person with missing measurements

        # Match detected people to existing tracks
        matched_tracks = self.match_detections_to_tracks_global_nearest_neighbour(propogated, detected_clusters)  
  
        # Update tracks with new oberservations 
        tracks_to_delete = set()   
        for idx, propogated_track in enumerate(propogated):
            track = self.objects_tracked[idx] # The corresponding non-propogated track
            if propogated_track in matched_tracks:
                matched_detection = matched_tracks[propogated_track]
                observations = np.array([matched_detection.pos_x, 
                                         matched_detection.pos_y])
                track.confidence = 0.95*track.confidence + 0.05*matched_detection.confidence                                                                         
                track.times_seen += 1
                track.last_seen = now
                track.seen_in_current_scan = True
            else: # propogated_track not matched to a detection
                observations = np.ma.masked_array(np.array([0, 0]), mask=[1,1]) # don't provide a measurement update for Kalman filter
                track.seen_in_current_scan = False
                        
            # Input observations to Kalman filter
            track.update(observations)


            # Check track for deletion because covariance is too large
            cov = track.filtered_state_covariances[0][0] + track.var_obs # cov_xx == cov_yy == cov
            if cov > self.max_cov:
                tracks_to_delete.add(track)                

        # Delete tracks that have been set for deletion
        for track in tracks_to_delete:         
            track.deleted = True # Because the tracks are also pointed to in self.potential_leg_pairs, we have to mark them deleted so they can deleted from that set too
            self.objects_tracked.remove(track)
            
        # If detections were not matched, create a new track  
        for detect in detected_clusters:      
            if not detect in matched_tracks.values():
                self.objects_tracked.append(ObjectTracked(detect.pos_x, detect.pos_y, now, detect.confidence))

        # Do some leg pairing to create potential people tracks/leg pairs
        for track_1 in self.objects_tracked:
            for track_2 in self.objects_tracked:
                if (track_1 != track_2 
                    and track_1.id_num > track_2.id_num 
                    and not track_1.person and not track_2.person 
                    and (track_1, track_2) not in self.potential_leg_pairs
                    ):
                    self.potential_leg_pairs.add((track_1, track_2))
                    self.potential_leg_pair_initial_dist_travelled[(track_1, track_2)] = (track_1.dist_travelled, track_2.dist_travelled)
        
        # We want to iterate over the potential leg pairs but iterating over the set <self.potential_leg_pairs> will produce arbitrary iteration orders.
        # This is bad if we want repeatable tests. Otherwise, it shouldn't affect performance.
        # So we'll create a sorted list and iterate over that.
        potential_leg_pairs_list = list(self.potential_leg_pairs)
        potential_leg_pairs_list.sort(key=lambda tup: (tup[0].id_num, tup[1].id_num))

        # Check if current leg pairs are still valid and if they should spawn a person
        leg_pairs_to_delete = set()   
        for track_1, track_2 in potential_leg_pairs_list:
            # Check if we should delete this pair because 
            # - the legs are too far apart 
            # - or one of the legs has already been paired 
            # - or a leg has been deleted because it hasn't been seen for a while
            dist = ((track_1.pos_x - track_2.pos_x)**2 + (track_1.pos_y - track_2.pos_y)**2)**(1./2.)
            if (dist > self.max_leg_pairing_dist 
                or track_1.person or track_2.person
                 or track_1.deleted or track_2.deleted 
                 or track_1.confidence < self.confidence_threshold_to_maintain_track 
                 or track_2.confidence < self.confidence_threshold_to_maintain_track
                 ):
                leg_pairs_to_delete.add((track_1, track_2))
                continue

            # Check if we should create a tracked person from this pair
			# Two conditions: 
			# - both tracks have been matched to a cluster in the current scan
			# - both tracks have travelled at least a distance of <self.dist_travelled_together_to_initiate_track_pair> since they were paired
            if track_1.seen_in_current_scan and track_2.seen_in_current_scan:
		        track_1_initial_dist, track_2_initial_dist = self.potential_leg_pair_initial_dist_travelled[(track_1, track_2)]
		        dist_travelled = min(track_1.dist_travelled - track_1_initial_dist, track_2.dist_travelled - track_2_initial_dist)
		        if dist_travelled > self.dist_travelled_together_to_initiate_leg_pair:
		            # Create a new person from this leg pair
		            self.people_tracked.append(TrackedPerson(track_1, track_2))
		            track_1.person = self.people_tracked[-1]
		            track_2.person = self.people_tracked[-1]
		            leg_pairs_to_delete.add((track_1, track_2))

        # Delete leg pairs set for deletion
        for leg_pair in leg_pairs_to_delete:
            self.potential_leg_pairs.remove(leg_pair)

        # Update tracked people
        people_to_delete = set()
        for person in self.people_tracked:
            # Remove references to tracks we want to delete
            if person.leg_1.deleted or person.leg_2.deleted:
                people_to_delete.add(person)
                continue
            # Check that legs haven't gotten too far apart or that have too low confidences
            # We use 2.*self.max_leg_pairing_dist as the max dist between legs before deleting because sometimes the legs will drift apart a bit then come back together when one is not seen
            dist = ((person.leg_1.pos_x - person.leg_2.pos_x)**2 + (person.leg_1.pos_y - person.leg_2.pos_y)**2)**(1./2.)
            if (dist > 2.*self.max_leg_pairing_dist 
                or person.leg_1.confidence < self.confidence_threshold_to_maintain_track 
                or person.leg_2.confidence < self.confidence_threshold_to_maintain_track
                ):
                people_to_delete.add(person)   

				# Purely for debugging:
                if (person.leg_1.confidence < self.confidence_threshold_to_maintain_track 
                    or person.leg_2.confidence < self.confidence_threshold_to_maintain_track
                    ):
                    rospy.loginfo("deleting due to low confidence")

        # Delete people set for deletion
        for person in people_to_delete:
            person.leg_1.person = None
            person.leg_2.person = None
            person.leg_1 = None
            person.leg_2 = None
            self.people_tracked.remove(person)

        self.publish_tracked_objects(now)
        self.publish_tracked_people(now)
            

    def publish_tracked_objects(self, now):
        """
        Publish markers of tracked objects to Rviz
        """        
        # Make sure we can get the required transform first:
        if self.use_scan_header_stamp_for_tfs:
            tf_time = now            
            try:
                self.listener.waitForTransform(self.publish_people_frame, self.fixed_frame, tf_time, rospy.Duration(1.0))
                transform_available = True
            except:
                transform_available = False
        else:
            tf_time = rospy.Time(0)
            transform_available = self.listener.canTransform(self.publish_people_frame, self.fixed_frame, tf_time)

        marker_id = 0
        if not transform_available:
            rospy.loginfo("Person tracker: tf not avaiable. Not publishing people")
        else:
            for track in self.objects_tracked:
                if self.publish_occluded or track.seen_in_current_scan: # Only publish people who have been seen in current scan, unless we want to publish occluded people
                    # Get the track position in the <self.publish_people_frame> frame
                    ps = PointStamped()
                    ps.header.frame_id = self.fixed_frame
                    ps.header.stamp = tf_time
                    ps.point.x = track.pos_x
                    ps.point.y = track.pos_y
                    try:
                        ps = self.listener.transformPoint(self.publish_people_frame, ps)
                    except:
                        continue

                    # publish rviz markers       
                    marker = Marker()
                    marker.header.frame_id = self.publish_people_frame
                    marker.header.stamp = now
                    marker.ns = "objects_tracked"
                    marker.color.r = track.colour[0]
                    marker.color.g = track.colour[1]
                    marker.color.b = track.colour[2]                                        
                    marker.color.a = 1
                    marker.pose.position.x = ps.point.x 
                    marker.pose.position.y = ps.point.y
                    marker.id = marker_id
                    marker_id += 1
                    marker.type = Marker.CYLINDER
                    marker.scale.x = 0.05
                    marker.scale.y = 0.05
                    marker.scale.z = 0.2
                    marker.pose.position.z = 0.15
                    self.marker_pub.publish(marker)

                    # # Publish a marker showing distance travelled:
                    # if track.dist_travelled > 1:
                    #     marker.color.r = 1.0
                    #     marker.color.g = 1.0
                    #     marker.color.b = 1.0
                    #     marker.color.a = 1.0
                    #     marker.id = marker_id
                    #     marker_id += 1
                    #     marker.type = Marker.TEXT_VIEW_FACING
                    #     marker.text = str(round(track.dist_travelled,1))
                    #     marker.scale.z = 0.1            
                    #     marker.pose.position.z = 0.6
                    #     self.marker_pub.publish(marker)      

                    # # Publish <self.confidence_percentile>% confidence bounds of person as an ellipse:
                    # cov = track.filtered_state_covariances[0][0] + track.var_obs # cov_xx == cov_yy == cov
                    # std = cov**(1./2.)
                    # gate_dist_euclid = scipy.stats.norm.ppf(1.0 - (1.0-self.confidence_percentile)/2., 0, std)                    
                    # marker.type = Marker.SPHERE
                    # marker.scale.x = 2*gate_dist_euclid
                    # marker.scale.y = 2*gate_dist_euclid
                    # marker.scale.z = 0.01   
                    # marker.color.r = track.colour[0]
                    # marker.color.g = track.colour[1]
                    # marker.color.b = track.colour[2]               
                    # marker.color.a = 0.1
                    # marker.pose.position.z = 0.0
                    # marker.id = marker_id 
                    # marker_id += 1                    
                    # self.marker_pub.publish(marker)

        # Clear previously published track markers
        for m_id in xrange(marker_id, self.prev_track_marker_id):
            marker = Marker()
            marker.header.stamp = now                
            marker.header.frame_id = self.publish_people_frame
            marker.ns = "objects_tracked"
            marker.id = m_id
            marker.action = marker.DELETE   
            self.marker_pub.publish(marker)
        self.prev_track_marker_id = marker_id



    def publish_tracked_people(self, now):
        """
        Publish markers of tracked people to Rviz and to <people_tracked> topic
        """          
        people_tracked_msg = PersonArray()
        people_tracked_msg.header.stamp = now
        people_tracked_msg.header.frame_id = self.publish_people_frame
        marker_id = 0

        # Make sure we can get the required transform first:
        if self.use_scan_header_stamp_for_tfs:
            tf_time = now
            try:
                self.listener.waitForTransform(self.publish_people_frame, self.fixed_frame, tf_time, rospy.Duration(1.0))
                transform_available = True
            except:
                transform_available = False
        else:
            tf_time = rospy.Time(0)
            transform_available = self.listener.canTransform(self.publish_people_frame, self.fixed_frame, tf_time)

        marker_id = 0
        if not transform_available:
            rospy.loginfo("Person tracker: tf not avaiable. Not publishing people")
        else:
            # Publish tracked people to /people_tracked topic and to rviz
            for person in self.people_tracked:
                leg_1 = person.leg_1
                leg_2 = person.leg_2
                if self.publish_occluded or leg_1.seen_in_current_scan or leg_2.seen_in_current_scan:
                    # Get person's position in the <self.publish_people_frame> frame 
                    ps = PointStamped()
                    ps.header.frame_id = self.fixed_frame
                    ps.header.stamp = tf_time
                    ps.point.x = (leg_1.pos_x + leg_2.pos_x)/2.
                    ps.point.y = (leg_1.pos_y + leg_2.pos_y)/2.
                    try:
                        ps = self.listener.transformPoint(self.publish_people_frame, ps)
                    except:
                        rospy.logerr("Not publishing people due to no transform from fixed_frame-->publish_people_frame")                                                                        
                        continue
                    
                    # publish to people_tracked topic
                    new_person = Person() 
                    new_person.pose.position.x = ps.point.x 
                    new_person.pose.position.y = ps.point.y 
                    new_person.id = person.id_num 
                    people_tracked_msg.people.append(new_person)

                    # publish rviz markers       
                    marker = Marker()
                    marker.header.frame_id = self.publish_people_frame
                    marker.header.stamp = now
                    marker.ns = "People_tracked"
                    marker.color.r = person.colour[0]
                    marker.color.g = person.colour[1]
                    marker.color.b = person.colour[2]                                      
                    marker.color.a = (rospy.Duration(3) - (rospy.get_rostime() - leg_1.last_seen)).to_sec()/rospy.Duration(3).to_sec() + 0.1
                    marker.pose.position.x = ps.point.x 
                    marker.pose.position.y = ps.point.y
                    for i in xrange(2): # publish two markers per person: one for body and one for head
                        marker.id = marker_id #person.id_num + 20000*i
                        marker_id += 1
                        if i==0: # cylinder for body shape
                            marker.type = Marker.CYLINDER
                            marker.scale.x = 0.2
                            marker.scale.y = 0.2
                            marker.scale.z = 1.2
                            marker.pose.position.z = 0.8
                        else: # sphere for head shape
                            marker.type = Marker.SPHERE
                            marker.scale.x = 0.2
                            marker.scale.y = 0.2
                            marker.scale.z = 0.2                
                            marker.pose.position.z = 1.5
                        self.marker_pub.publish(marker)    
                    # Text showing person's ID number 
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 1.0
                    marker.color.a = 1.0
                    marker.id = marker_id
                    marker_id += 1
                    marker.type = Marker.TEXT_VIEW_FACING
                    marker.text = str(person.id_num)
                    marker.scale.z = 0.2         
                    marker.pose.position.z = 1.7
                    self.marker_pub.publish(marker)                          

        # Clear previously published people markers
        for m_id in xrange(marker_id, self.prev_person_marker_id):
            marker = Marker()
            marker.header.stamp = now                
            marker.header.frame_id = self.publish_people_frame
            marker.ns = "People_tracked"
            marker.id = m_id
            marker.action = marker.DELETE   
            self.marker_pub.publish(marker)
        self.prev_person_marker_id = marker_id          

        # Publish people tracked message
        self.people_tracked_pub.publish(people_tracked_msg)            


if __name__ == '__main__':
    rospy.init_node('multi_person_tracker', anonymous=True)
    kmt = KalmanMultiTracker()





