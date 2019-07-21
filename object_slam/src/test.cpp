#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

#include <quadric_slam/Frame.h>
#include <quadric_slam/Object_landmark.h>
#include <quadric_slam/g2o_Object.h>

using namespace g2o;
int main() {
  std::string base_folder =
      ros::package::getPath("object_slam") + "/data/quadic_test_data/";
  Matrix3d calib;
  calib << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;

  Quadric quadric = Quadric();
  Eigen::MatrixXd rawQuadric(4, 19);
  read_all_number_txt(base_folder + "quadric_parameters.txt", rawQuadric);

  Eigen::MatrixXd rawCamPose(35, 16);
  read_all_number_txt(base_folder + "camera_rt35.txt", rawCamPose);

  std::cout << "rawQuadric \n" << rawQuadric << std::endl << std::endl;
  std::cout << "rawCamPose \n" << rawCamPose << std::endl;
  // set quadric
  quadric.scale = rawQuadric.block(0, 0, 1, 3);
  Eigen::MatrixXd Rq(3, 3);
  Rq.block(0, 0, 1, 3) = rawQuadric.block(0, 3, 1, 3);
  Rq.block(1, 0, 1, 3) = rawQuadric.block(0, 7, 1, 3);
  Rq.block(2, 0, 1, 3) = rawQuadric.block(0, 11, 1, 3);
  std::cout << "Rq" << Rq << std::endl;

  Vector3d Tq = rawQuadric.block(0, 15, 1, 3);
  std::cout << "Tq" << Tq << std::endl;

  quadric.pose = SE3Quat(Rq, Tq);

  // set cam pose
  Eigen::MatrixXd Rc(3, 3);
  Rc.block(0, 0, 1, 3) = rawCamPose.block(0, 0, 1, 3);
  Rc.block(1, 0, 1, 3) = rawCamPose.block(0, 4, 1, 3);
  Rc.block(2, 0, 1, 3) = rawCamPose.block(0, 8, 1, 3);
  Vector3d Tc = rawQuadric.block(0, 12, 1, 3);
  std::cout << "Rc" << Rc << std::endl;
  std::cout << "Tc" << Tc << std::endl;
  SE3Quat camPose = SE3Quat(Rc, Tc);

  Vector4d bbox = quadric.projectOntoImageRect(camPose.inverse(), calib);
  cv::Mat src = imread(base_folder + "1311867170.462290.png");

  cv::rectangle(src, Point(bbox(0), bbox(1)), Point(bbox(2), bbox(3)),
                Scalar(255, 0, 0), 1, LINE_8, 0);

  cv::imshow("src", src);
  cv::waitKey(0);

  return 0;
}
