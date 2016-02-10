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

#include <leg_tracker/laser_processor.h>


namespace laser_processor
{

Sample* Sample::Extract(int ind, const sensor_msgs::LaserScan& scan)
{
  Sample* s = new Sample();

  s->index = ind;
  s->range = scan.ranges[ind];
  s->x = cos( scan.angle_min + ind*scan.angle_increment ) * s->range;
  s->y = sin( scan.angle_min + ind*scan.angle_increment ) * s->range;
  if (s->range > scan.range_min && s->range < scan.range_max)
  {
    return s;
  }
  else
  {
    delete s;
    return NULL;
  }
}

void SampleSet::clear()
{
  for (SampleSet::iterator i = begin(); i != end(); ++i)
    delete (*i);
  std::set<Sample*, CompareSample>::clear();
}


tf::Point SampleSet::getPosition()
{
  float x_mean = 0.0;
  float y_mean = 0.0;
  for (iterator i = begin(); i != end(); ++i)
  {
    x_mean += ((*i)->x)/size();
    y_mean += ((*i)->y)/size();
  }

  return tf::Point (x_mean, y_mean, 0.0);
}


ScanProcessor::ScanProcessor(const sensor_msgs::LaserScan& scan) 
{
  scan_ = scan;

  SampleSet* cluster = new SampleSet;

  for (int i = 0; i < scan.ranges.size(); i++)
  {
    Sample* s = Sample::Extract(i, scan);

    if (s != NULL)
      cluster->insert(s);
  }

  clusters_.push_back(cluster);
}

ScanProcessor::~ScanProcessor()
{
  for ( std::list<SampleSet*>::iterator c = clusters_.begin(); c != clusters_.end(); ++c)
    delete (*c);
}

void ScanProcessor::removeLessThan(uint32_t num)
{
  std::list<SampleSet*>::iterator c_iter = clusters_.begin();
  while (c_iter != clusters_.end())
  {
    if ( (*c_iter)->size() < num )
    {
      delete (*c_iter);
      clusters_.erase(c_iter++);
    } 
    else 
    {
      ++c_iter;
    }
  }
}


void ScanProcessor::splitConnected(float thresh)
{
  // Holds our temporary list of split clusters 
  // because we will be modifying our existing list in the mean time
  std::list<SampleSet*> tmp_clusters;

  std::list<SampleSet*>::iterator c_iter = clusters_.begin();

  while (c_iter != clusters_.end())
  {
    while ((*c_iter)->size() > 0)
    {
      // Iterate over laser scan samples in clusters_
      // and collect those which are within a euclidian distance of <thresh>
      // and store new clusters in tmp_clusters
      SampleSet::iterator s_first = (*c_iter)->begin();
      std::list<Sample*> sample_queue;
      sample_queue.push_back(*s_first);
      (*c_iter)->erase(s_first);
      std::list<Sample*>::iterator s_q = sample_queue.begin();
      while (s_q != sample_queue.end())
      {
        int expand = (int)(asin( thresh / (*s_q)->range ) / scan_.angle_increment);

        SampleSet::iterator s_rest = (*c_iter)->begin();

        while ( (s_rest != (*c_iter)->end() and (*s_rest)->index < (*s_q)->index + expand ) )
        {
          if (sqrt( pow( (*s_q)->x - (*s_rest)->x, 2.0f) + pow( (*s_q)->y - (*s_rest)->y, 2.0f)) < thresh)
          {
            sample_queue.push_back(*s_rest);
            (*c_iter)->erase(s_rest++);
          } 
          else 
          {
            ++s_rest;
          }  
        }
        s_q++;
      }

      // Move all the samples into the new cluster
      SampleSet* c = new SampleSet;
      for (s_q = sample_queue.begin(); s_q != sample_queue.end(); s_q++)
        c->insert(*s_q);

      // Store the temporary clusters
      tmp_clusters.push_back(c);
    }

    //Now that c_iter is empty, we can delete
    delete (*c_iter);

    //And remove from the map
    clusters_.erase(c_iter++);
  }

  // Insert our temporary clusters list back into the de facto list
  clusters_.insert(clusters_.begin(), tmp_clusters.begin(), tmp_clusters.end());
}
}; // namespace laser_processor 
