/*********************************************************************
* Software License Agreement (BSD License)
* 
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
* 
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
* 
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
* 
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>

#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/ml.h>

#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseArray.h>

#include <leg_tracker/laser_processor.h>
#include <leg_tracker/cluster_features.h>



/**
* @brief Trains the leg_detector to classify scan clusters as legs/not legs
* 
* Reads in rosbags: 
*   positive examples are obtained from rosbags annotated by extract_positive_training_clusters 
*   negative examples are obtained from rosbags where the laser was moving around an empty room
* Output is an .yaml file which configures an OpenCV random forest classifier
*/
class TrainLegDetector
{
public: 
  /**
  * @brief Constructor 
  */
  TrainLegDetector(ros::NodeHandle nh): 
    feat_count_(0)
  {
    // Get ROS params (all have default values so it's not critical we get them all)
    nh.param("cluster_dist_euclid", cluster_dist_euclid_, 0.13);
    nh.param("min_points_per_cluster", min_points_per_cluster_, 3);
    nh.param("undersample_negative_factor", undersample_negative_factor_, 50);
    nh.param("positive_leg_cluster_positions_topic", positive_leg_cluster_positions_topic_, std::string("/leg_cluster_positions"));

    // Print back params:
    printf("\nROS parameters: \n");
    printf("cluster_dist_euclid:%.2fm \n", cluster_dist_euclid_);
    printf("min_points_per_cluster:%i \n", min_points_per_cluster_);
    printf("undersample_negative_factor:%i \n", undersample_negative_factor_);
    printf("positive_leg_cluster_positions_topic:%s \n", positive_leg_cluster_positions_topic_.c_str());
    printf("\n");
  }


  /**
  * @brief Prase command line arguments and load training and test data
  * @params argc Command line arguments
  * @params argv Command line arguments
  */
  void loadData(int argc, char **argv)
  {
    // Parse command line arguements and load data 
    printf("\nLoading data...\n");
    for (int i = 0; i < argc; i++) 
    {
      if (!strcmp(argv[i],"--pos"))
      {
        char* rosbag_file = argv[++i]; 
        char* scan_topic = argv[++i];
        loadPosData(rosbag_file, scan_topic, train_pos_data_);
      }
      else if (!strcmp(argv[i],"--neg"))
      {
        char* rosbag_file = argv[++i]; 
        char* scan_topic = argv[++i];
        loadNegData(rosbag_file, scan_topic, train_neg_data_);
      }
      else if (!strcmp(argv[i],"--test_pos"))
      {
        char* rosbag_file = argv[++i]; 
        char* scan_topic = argv[++i];
        loadPosData(rosbag_file, scan_topic, test_pos_data_);
      }
      else if (!strcmp(argv[i],"--test_neg"))
      {
        char* rosbag_file = argv[++i]; 
        char* scan_topic = argv[++i];
        loadNegData(rosbag_file, scan_topic, test_neg_data_);
      }
      else if (!strcmp(argv[i],"--save_file"))
      {
        save_file_ = std::string(argv[++i]);
      }
    }

    // Check we have a valid save file
    if (save_file_.empty())
    {
      ROS_ERROR("Save file not specified properly in command line arguments \nExiting");
      exit(1);
    }

    // Error check the loaded data
    if (train_pos_data_.empty() or train_neg_data_.empty()) 
    {
      ROS_ERROR("Data not loaded from rosbags properly \nExiting");
      exit(1);
    }

    printf("\n  Total positive training samples: %i \t Total negative training samples: %i \n", (int)train_pos_data_.size(), (int)train_neg_data_.size());
    printf("  Total positive test samples: %i \t Total negative test samples: %i \n\n", (int)test_pos_data_.size(), (int)test_neg_data_.size());
  }


  /**
  * @brief Train the classifier using the positive and negative training data
  *
  * We will use a OpenCV's random forest classifier to do the training.
  * Results will be saved in the forest_ variable. 
  */
  void train()
  {
    int sample_size = train_pos_data_.size() + train_neg_data_.size();
    feat_count_ = train_pos_data_[0].size();

    CvMat* cv_data = cvCreateMat( sample_size, feat_count_, CV_32FC1);
    CvMat* cv_resp = cvCreateMat( sample_size, 1, CV_32S);

    // Put positive data in opencv format.
    int j = 0;
    for (std::vector< std::vector<float> >::const_iterator i = train_pos_data_.begin();
         i != train_pos_data_.end();
         i++)
    {
      float* data_row = (float*)(cv_data->data.ptr + cv_data->step*j);
      for (int k = 0; k < feat_count_; k++)
        data_row[k] = (*i)[k];
      
      cv_resp->data.i[j] = 1;
      j++;
    }

    // Put negative data in opencv format.
    for (std::vector< std::vector<float> >::const_iterator i = train_neg_data_.begin();
         i != train_neg_data_.end();
         i++)
    {
      float* data_row = (float*)(cv_data->data.ptr + cv_data->step*j);
      for (int k = 0; k < feat_count_; k++)
        data_row[k] = (*i)[k];
      
      cv_resp->data.i[j] = -1;
      j++;
    }

    CvMat* var_type = cvCreateMat( 1, feat_count_ + 1, CV_8U );
    cvSet( var_type, cvScalarAll(CV_VAR_ORDERED));
    cvSetReal1D( var_type, feat_count_, CV_VAR_CATEGORICAL );
    
    // Random forest training parameters
    // One important parameter not set here is undersample_negative_factor.
    // I tried to keep the params similar to the defaults in scikit-learn
    float priors[] = {1.0, 1.0};

    CvRTParams fparam(
      10000,              // max depth of tree
      2,                  // min sample count to split tree
      0,                  // regression accuracy (?)
      false,              // use surrogates (?)
      1000,               // max categories
      priors,             // priors
      false,              // calculate variable importance 
      2,                  // number of active vars for each tree node (default from scikit-learn is: (int)round(sqrt(feat_count_))
      100,                // max trees in forest (default of 10 from scikit-learn does worse)
      0.001f,             // forest accuracy (sufficient OOB error)
      CV_TERMCRIT_ITER    // termination criteria. CV_TERMCRIT_ITER = once we reach max number of forests
      ); 
       
    forest_.train( 
      cv_data,                // train data 
      CV_ROW_SAMPLE,          // tflag
      cv_resp,                // responses (i.e. labels)
      0,                      // varldx (?)
      0,                      // sampleldx (?)
      var_type,               // variable type 
      0,                      // missing data mask
      fparam                  // parameters 
      );                

    cvReleaseMat(&cv_data);
    cvReleaseMat(&cv_resp);
    cvReleaseMat(&var_type);
  }


  /**
  * @brief Test the classifier using pos_data and neg_data
  * @params on_training_set If we should use training set instead of test set
  */
  void test(
    bool on_training_set = false
    )
  {
    std::vector< std::vector<float> > pos_data;
    std::vector< std::vector<float> > neg_data;
    if (on_training_set)
    {
      pos_data = train_pos_data_;
      neg_data = train_neg_data_;
    }
    else
    {
      pos_data = test_pos_data_;
      neg_data = test_neg_data_;
    }

    int correct_pos = 0;
    int correct_neg = 0;

    CvMat* tmp_mat = cvCreateMat(1,feat_count_,CV_32FC1);

    // test on positive examples
    for (std::vector< std::vector<float> >::const_iterator i = pos_data.begin();
         i != pos_data.end();
         i++)
    {
      for (int k = 0; k < feat_count_; k++)
        tmp_mat->data.fl[k] = (float)((*i)[k]);
      if (forest_.predict( tmp_mat) > 0)
        correct_pos++;
    }

    // test on negative examples
    for (std::vector< std::vector<float> >::const_iterator i = neg_data.begin();
         i != neg_data.end();
         i++)
    {
      for (int k = 0; k < feat_count_; k++)
        tmp_mat->data.fl[k] = (float)((*i)[k]);
      if (forest_.predict( tmp_mat ) < 0)
        correct_neg++;
    }

    printf("   Positive: %d/%d \t\t Error: %g%%\n", correct_pos, (int)pos_data.size(), 100.0 - 100.0*(float)(correct_pos)/(int)pos_data.size());
    printf("   Negative: %d/%d \t\t Error: %g%%\n", correct_neg, (int)neg_data.size(), 100.0 - 100.0*(float)(correct_neg)/(int)neg_data.size());
    printf("   Combined: %d/%d \t\t Error: %g%%\n\n", correct_pos + correct_neg, (int)pos_data.size() + (int)neg_data.size(), 100.0 - 100.0*(float)(correct_pos + correct_neg)/((int)pos_data.size() + (int)neg_data.size()));

    cvReleaseMat(&tmp_mat);    
  }


  /**
  * @brief Save the OpenCV random forest configuration as a yaml file
  */
  void save()
  {
    printf("Saving classifier as: %s\n", save_file_.c_str());    
    forest_.save(save_file_.c_str());
  }


private:
  CvRTrees forest_;

  int feat_count_;

  ClusterFeatures cf_;

  double cluster_dist_euclid_;
  int min_points_per_cluster_;  
  int undersample_negative_factor_;
  std::string positive_leg_cluster_positions_topic_;

  std::vector< std::vector<float> > train_pos_data_;
  std::vector< std::vector<float> > train_neg_data_;
  std::vector< std::vector<float> > test_neg_data_;
  std::vector< std::vector<float> > test_pos_data_;
  std::string save_file_;

  /**
  * @brief Load positive training data from a rosbag
  * @params rosbag_file ROS bag to load data from
  * @params scan_topic Scan topic we should draw the data from in the ROS bag
  * @params data All loaded data is returned in this var
  * 
  * Separate the scan into clusters, figure out which clusters lie near a positive marker,
  * calcualte features on those clusters, save features from each cluster to <data>.
  */
  void loadPosData(
    const char* rosbag_file, 
    const char* scan_topic, 
    std::vector< std::vector<float> > &data
    )
  {
    rosbag::Bag bag;
    bag.open(rosbag_file, rosbag::bagmode::Read);
    std::vector<std::string> topics;
    topics.push_back(std::string(scan_topic)); 
    topics.push_back(std::string(positive_leg_cluster_positions_topic_)); 
    rosbag::View view(bag, rosbag::TopicQuery(topics)); 

    geometry_msgs::PoseArray positive_clusters;

    int message_num = 0;
    int initial_pos_data_size = (int)data.size();
    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {
      geometry_msgs::PoseArray::ConstPtr pose_array_msg = m.instantiate<geometry_msgs::PoseArray>();
      if (pose_array_msg != NULL)
      {
        positive_clusters = *pose_array_msg;
      }

      sensor_msgs::LaserScan::ConstPtr scan = m.instantiate<sensor_msgs::LaserScan>();
      if (scan != NULL and positive_clusters.poses.size())
      {  
        laser_processor::ScanProcessor processor(*scan);
        processor.splitConnected(cluster_dist_euclid_);
        processor.removeLessThan(min_points_per_cluster_);
 
        for (std::list<laser_processor::SampleSet*>::iterator i = processor.getClusters().begin();
             i != processor.getClusters().end();
             i++)
        {
          tf::Point cluster_position = (*i)->getPosition();

          for (int j = 0; 
                   j < positive_clusters.poses.size();
                   j++)
          {
            // Only use clusters which are close to a "marker"
            double dist_x = positive_clusters.poses[j].position.x - cluster_position[0],
                   dist_y = positive_clusters.poses[j].position.y - cluster_position[1],             
                   dist_abs = sqrt(dist_x*dist_x + dist_y*dist_y);
            if (dist_abs < 0.0001)
            {
              data.push_back(cf_.calcClusterFeatures(*i, *scan));
              break;                                           
            }
          }
        }
        message_num++;
      } 
    }
    bag.close();

    printf("\t Got %i scan messages, %i samples, from %s  \n",message_num, (int)data.size() - initial_pos_data_size, rosbag_file);
  } 


  /**
  * @brief Load negative training data from a rosbag
  * @params rosbag_file ROS bag to load data from
  * @params scan_topic Scan topic we should draw the data from in the ROS bag
  * @params data All loaded data is returned in this var
  * 
  *  Load scan messages from the rosbag_file, separate into clusters, 
  * calcualte features on those clusters, save features from each cluster to <data>.
  */
  void loadNegData(
    const char* rosbag_file, 
    const char* scan_topic, 
    std::vector< std::vector<float> > &data
    )
  {
    rosbag::Bag bag;
    bag.open(rosbag_file, rosbag::bagmode::Read);
    std::vector<std::string> topics;
    topics.push_back(std::string(scan_topic));
    rosbag::View view(bag, rosbag::TopicQuery(topics)); 

    int message_num = 0;
    int initial_neg_data_size = (int)data.size();
    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {
      sensor_msgs::LaserScan::ConstPtr scan = m.instantiate<sensor_msgs::LaserScan>();
      if (scan != NULL)
      {
        laser_processor::ScanProcessor processor(*scan);
        processor.splitConnected(cluster_dist_euclid_);
        processor.removeLessThan(min_points_per_cluster_);
 
        for (std::list<laser_processor::SampleSet*>::iterator i = processor.getClusters().begin();
             i != processor.getClusters().end();
             i++)
        {
          if (rand() % undersample_negative_factor_ == 0) // one way of undersampling the negative class
            data.push_back(cf_.calcClusterFeatures(*i, *scan));                 
        }
        message_num++;
      } 
    }
    bag.close();

    printf("\t Got %i scan messages, %i samples, from %s  \n",message_num, (int)data.size() - initial_neg_data_size, rosbag_file);
  } 
};


int main(int argc, char **argv)
{
  // Declare a ROS node so we can get ROS parameters from the server
  ros::init(argc, argv,"train_leg_detector");
  ros::NodeHandle nh;

  TrainLegDetector tld(nh);
  tld.loadData(argc, argv);

  // Training 
  printf("Training classifier...");
  tld.train();
  printf("done! \n\n");

  // Testing
  printf("Testing classifier...\n");
  printf(" training set: \n");
  tld.test(true);    // Test on training set
  printf(" test set: \n");
  tld.test(false);    // Test on test set

  // Save .yaml file
  tld.save();

  printf("Finished successfully!\n");
  return 0;
}
