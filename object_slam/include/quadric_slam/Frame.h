#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "quadric_slam/Object_landmark.h"
#include "quadric_slam/g2o_Object.h"
class Quadric_landmark;

class tracking_frame_quadric {
 public:
  const int frame_seq_id;  // image topic sequence id, fixed
  cv::Mat frame_img;
  cv::Mat quadrics_2d_img;

  std::shared_ptr<g2o::VertexSE3Expmap> pose_vertex;

  std::vector<std::shared_ptr<Detection_result>>
      detect_results;         // object detection result
  g2o::SE3Quat cam_pose_Tcw;  // optimized pose  world to cam
  g2o::SE3Quat cam_pose_Twc;  // optimized pose  cam to world
  tracking_frame_quadric(int id) : frame_seq_id(id) {}
};
