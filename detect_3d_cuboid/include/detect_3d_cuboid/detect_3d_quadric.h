#pragma once

// std c
#include <string>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

//OpenCV
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>

//g2o
#include "../../object_slam/Thirdparty/g2o"

//
#include "detect_3d_cuboid/matrix_utils.h"




struct cam_pose_infos
{
      Eigen::Matrix4d transToWolrd;
      Eigen::Matrix3d Kalib;
      
      Eigen::Matrix3d rotationToWorld;
      Eigen::Vector3d euler_angle;
      Eigen::Matrix3d invR;
      Eigen::Matrix3d invK;
      Eigen::Matrix<double, 3, 4> projectionMatrix;      
      Eigen::Matrix3d KinvR; // K*invR
      double camera_yaw;
};


class detect_3d_quadric
{
public:
      cam_pose_infos cam_pose;
      cam_pose_infos cam_pose_raw;
      void set_calibration(const Eigen::Matrix3d& Kalib);
      void set_cam_pose(const Eigen::Matrix4d& transToWolrd);

      // object detector needs image, camera pose, and 2D bounding boxes(n*5, each row: xywh+prob)  long edges: n*4.  all number start from 0
      void detect_quadric(const cv::Mat& rgb_img, const Eigen::Matrix4d& transToWolrd,const Eigen::MatrixXd& obj_bbox_coors, Eigen::MatrixXd edges,
			 std::vector<ObjectSet>& all_object_cuboids);      
            
      bool whether_plot_detail_images = false;
      bool whether_plot_final_images = false;
      bool whether_save_final_images = false; cv::Mat cuboids_2d_img;  // save to this opencv mat
      bool print_details = false;
      
      // important mode parameters for proposal generation.
      bool consider_config_1 = true;  // false true
      bool consider_config_2 = true;
      bool whether_sample_cam_roll_pitch = false; // sample camera roll pitch in case don't have good camera pose
      bool whether_sample_bbox_height = false;  // sample object height as raw detection might not be accurate
      

      int max_cuboid_num = 1;  	      //final return best N cuboids
      double nominal_skew_ratio = 1;  // normally this 1, unless there is priors
      double max_cut_skew = 3;
};

typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;
using namespace g2o{
class Quadric{
Public:
 SE3Quat pose;
 Vector3d scale;   //semi-axis a,b,c
 Quadric()
{
Pose=SE3Quat();
Scale.setZero();
}
inline void fromMinimalVector(const Vector9d& v){
Eigen::Quaterniond posequat = zyx_euler_to_quat(v(3),v(4),v(5));
pose = SE3Quat(posequat, v.head<3>());
scale = v.tail<3>();
 }
// xyz quaternion,
 inline void fromVector(const Vector10d& v){
pose.fromVector(v.head<7>());
scale = v.tail<3>();
 }
inline const Vector3d& translation() const {return pose.translation();}
 inline void setTranslation(const Vector3d& t_) {pose.setTranslation(t_);}
 inline void setRotation(const Quaterniond& r_) {pose.setRotation(r_);}
 inline void setRotation(const Matrix3d& R) {pose.setRotation(Quaterniond(R));}
 inline void setScale(const Vector3d &scale_) {scale=scale_;}

 // apply update to current quadric, exponential map
Quadric exp_update(const Vector9d& update){
Qaudric res;
          res.pose = this->pose*SE3Quat::exp(update.head<6>());
res.scale = this->scale + update.tail<3>();
          return res;
 }
}



}
typedef std::vector<Quadric*> ObjectSet;  // for each 2D box, the set of generated 3D cuboids


