#pragma once

#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "detect_3d_cuboid/matrix_utils.h"

#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/StdVector>

#include <algorithm>  // std::swap

#define WIDTH 640
#define HEIGHT 480

typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;
typedef Eigen::Matrix<double, 5, 1> Vector5d;
namespace g2o {

class Quadric {
 public:
  SE3Quat pose;
  Vector3d scale;  // semi-axis a,b,c
  Vector5d bbox;   // [center.x center.y width height pro]

  Quadric() {
    pose = SE3Quat();
    scale.setZero();
  }

  // v = (t1,t2,t3,theta1,theta2,theta3,s1,s2,s3)
  // xyz roll pitch yaw half_scale
  inline void fromMinimalVector(const Vector9d& v) {
    Eigen::Quaterniond posequat = zyx_euler_to_quat(v(3), v(4), v(5));
    pose = SE3Quat(posequat, v.head<3>());
    scale = v.tail<3>();
  }
  // xyz quaternion,
  inline void fromVector(const Vector10d& v) {
    Eigen::Matrix4d dual_quadric, raw_quadric;
    Eigen::Vector3d rotation;
    Eigen::Vector3d shape;
    Eigen::Vector3d translation;

    dual_quadric << v(0, 0), v(1, 0), v(2, 0), v(3, 0), v(1, 0), v(4, 0),
        v(5, 0), v(6, 0), v(2, 0), v(5, 0), v(7, 0), v(8, 0), v(3, 0), v(6, 0),
        v(8, 0), v(9, 0);

    raw_quadric = dual_quadric.inverse() * cbrt(dual_quadric.determinant());
    //    cout << "The Primary Form of Quadric is " << endl
    //         << raw_quadric << endl
    //         << endl;

    Eigen::Matrix3d quadric_33;
    quadric_33 = raw_quadric.block(0, 0, 3, 3);

    double det = raw_quadric.determinant() / quadric_33.determinant();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(quadric_33);

    Eigen::Matrix3d eigen_vectors = eigen_solver.eigenvectors();
    //    cout << "The Rotation of the Constrained Quadric is " << endl
    //         << eigen_vectors << endl
    //         << endl;

    rotation = eigen_vectors.eulerAngles(2, 1, 0);
    //    cout << "Euler Angle is " << endl << rotation << endl << endl;
    Eigen::Vector3d eigen_values;
    eigen_values = eigen_solver.eigenvalues();

    Eigen::Vector3d eigen_values_inverse;
    eigen_values_inverse = eigen_values.array().inverse();

    shape << (((-det) * eigen_values_inverse).array().abs()).array().sqrt();
    //    cout << "The Shape of the Constrained Quadric is " << endl
    //         << shape << endl
    //         << endl;

    translation << v(3, 0) / v(9, 0), v(6, 0) / v(9, 0), v(8, 0) / v(9, 0);
    //    cout << "The Translation of the Constrained Quadric is " << endl
    //         << translation << endl
    //         << endl;
    Eigen::Quaterniond posequat = zyx_euler_to_quat(
        rotation(0, 0), rotation(1, 0), rotation(2, 0));  // may be xyz_euler

    pose = SE3Quat(posequat, translation.head<3>());
    scale = shape.head<3>();
  }

  inline const Vector3d& translation() const { return pose.translation(); }
  inline void setTranslation(const Vector3d& t_) { pose.setTranslation(t_); }
  inline void setRotation(const Quaterniond& r_) { pose.setRotation(r_); }
  inline void setRotation(const Matrix3d& R) {
    pose.setRotation(Quaterniond(R));
  }
  inline void setScale(const Vector3d& scale_) { scale = scale_; }

  // apply update to current quadric, exponential map
  Quadric exp_update(const Vector9d& update) {
    Quadric res;
    res.pose = this->pose * SE3Quat::exp(update.head<6>());
    res.scale = this->scale + update.tail<3>();
    return res;
  }

  // actual error between two qudrics.
  Vector9d quadric_log_error(const Quadric& newone) const {
    Vector9d res;
    SE3Quat pose_diff = newone.pose.inverse() * this->pose;
    res.head<6>() =
        pose_diff
            .log();  // treat as se3 log error. could also just use yaw error
    res.tail<3>() = this->scale - newone.scale;
    return res;
  }

  // Todo
  // function called by g2o.
  Vector9d min_log_error(const Quadric& newone,
                         bool print_details = false) const {
    bool whether_rotate_cubes =
        true;  // whether rotate cube to find smallest error
    if (!whether_rotate_cubes)
      return quadric_log_error(newone);

    // NOTE rotating cuboid... since we cannot determine the front face
    // consistenly, different front faces indicate different yaw, scale
    // representation. need to rotate all 360 degrees (global cube might be
    // quite different from local cube) this requires the sequential object
    // insertion. In this case, object yaw practically should not change much.
    // If we observe a jump, we can use code here to adjust the yaw.
    Vector4d rotate_errors_norm;
    Vector4d rotate_angles(-1, 0, 1, 2);  // rotate -90 0 90 180
    Eigen::Matrix<double, 9, 4> rotate_errors;
    for (int i = 0; i < rotate_errors_norm.rows(); i++) {
      Quadric rotated_quadric = newone.rotate_quadric(
          rotate_angles(i) * M_PI / 2.0);  // rotate new cuboids
      Vector9d quadric_error = this->quadric_log_error(rotated_quadric);
      rotate_errors_norm(i) = quadric_error.norm();
      rotate_errors.col(i) = quadric_error;
    }
    int min_label;
    rotate_errors_norm.minCoeff(&min_label);
    if (print_details)
      if (min_label != 1)
        std::cout << "Rotate Quadric   " << min_label << std::endl;
    return rotate_errors.col(min_label);
  }

  // change front face by rotate along current body z axis. another way of
  // representing cuboid. representing same cuboid (IOU always 1)
  Quadric rotate_quadric(double yaw_angle)
      const  // to deal with different front surface of cuboids
  {
    Quadric res;
    SE3Quat rot(
        Eigen::Quaterniond(cos(yaw_angle * 0.5), 0, 0, sin(yaw_angle * 0.5)),
        Vector3d(0, 0, 0));  // change yaw to rotation.
    res.pose = this->pose * rot;
    res.scale = this->scale;
    if ((yaw_angle == M_PI / 2.0) || (yaw_angle == -M_PI / 2.0) ||
        (yaw_angle == 3 * M_PI / 2.0))
      std::swap(res.scale(0), res.scale(1));

    return res;
  }

  // transform a local cuboid to global cuboid  Twc is camera pose. from camera
  // to world
  Quadric transform_from(const SE3Quat& Twc) const {
    Quadric res;
    res.pose = Twc * this->pose;
    res.scale = this->scale;
    return res;
  }

  // transform a global cuboid to local cuboid  Twc is camera pose. from camera
  // to world
  Quadric transform_to(const SE3Quat& Twc) const {
    Quadric res;
    res.pose = Twc.inverse() * this->pose;
    res.scale = this->scale;
    return res;
  }

  // xyz roll pitch yaw half_scale
  inline Vector9d toMinimalVector() const {
    Vector9d v;
    v.head<6>() = pose.toXYZPRYVector();
    v.tail<3>() = scale;
    return v;
  }

  // xyz quaternion, half_scale
  inline Vector10d toVector() const {
    Vector10d v;
    // Toto
    return v;
  }

  Matrix4d toSymMat() const {
    Matrix4d res;
    Matrix4d centreAtOrigin;
    centreAtOrigin = Eigen::Matrix4d::Identity();
    centreAtOrigin(0, 0) = pow(scale(0), 2);
    centreAtOrigin(1, 1) = pow(scale(1), 2);
    centreAtOrigin(2, 2) = pow(scale(2), 2);
    centreAtOrigin(3, 4) = 1;
    Matrix4d Z;
    Z = pose.to_homogeneous_matrix();
    res = Z * centreAtOrigin * Z.transpose();
    return res;
  }

  // get rectangles after projection  [topleft, bottomright]

  Matrix3d toConic(const SE3Quat& campose_cw, const Matrix3d& Kalib) const {
    Eigen::Matrix<double, 3, 4> P =
        Kalib * campose_cw.to_homogeneous_matrix().block(0, 0, 3, 4);
    Matrix4d symMat = this->toSymMat();
    Matrix3d conic = P * symMat * P.transpose();
    return conic;
  }

  Vector4d projectOntoImageRect(const SE3Quat& campose_cw,
                                const Matrix3d& Kalib) const {
    Matrix3d conic = this->toConic(campose_cw, Kalib);
    Vector6d c;
    c << conic(0, 0), conic(0, 1) / 2, conic(1, 1), conic(0, 2) / 2,
        conic(1, 2) / 2, conic(2, 2);
    Vector2d y, x;
    y(0) = 4 * c(4) - 2 * c(1) * c(3) +
           sqrt(pow(2 * c(1) * c(3) - 4 * c(4), 2) -
                4 * pow(c(1), 2) * (pow(c(3), 2) - 4 * c(5))) /
               (2 * (pow(c(1), 2) - 4 * c(2)));
    y(1) = 4 * c(4) - 2 * c(1) * c(3) -
           sqrt(pow(2 * c(1) * c(3) - 4 * c(4), 2) -
                4 * pow(c(1), 2) * (pow(c(3), 2) - 4 * c(5))) /
               (2 * (pow(c(1), 2) - 4 * c(2)));
    x(0) = 4 * c(3) - 2 * c(1) * c(4) +
           sqrt(pow(2 * c(1) * c(4) - 4 * c(3), 2) -
                4 * pow(c(1), 2) * (pow(c(4), 2) - 4 * c(5))) /
               (2 * (pow(c(1), 2) - 4 * c(0)));
    x(1) = 4 * c(3) - 2 * c(1) * c(4) -
           sqrt(pow(2 * c(1) * c(4) - 4 * c(3), 2) -
                4 * pow(c(1), 2) * (pow(c(4), 2) - 4 * c(5))) /
               (2 * (pow(c(1), 2) - 4 * c(0)));
    Vector2d bottomright;  // x y
    Vector2d topleft;
    bottomright(0) = x.maxCoeff();
    bottomright(1) = y.maxCoeff();
    topleft(0) = x.minCoeff();
    topleft(1) = y.minCoeff();
    // Todo:conic at boundary
    return Vector4d(topleft(0), topleft(1), bottomright(0), bottomright(1));
  }

  // get rectangles after projection  [center, width, height]
  Vector4d projectOntoImageBbox(const SE3Quat& campose_cw,
                                const Matrix3d& Kalib) const {
    Vector4d rect_project = projectOntoImageRect(
        campose_cw, Kalib);  // top_left, bottom_right  x1 y1 x2 y2
    Vector2d rect_center =
        (rect_project.tail<2>() + rect_project.head<2>()) / 2;
    Vector2d widthheight = rect_project.tail<2>() - rect_project.head<2>();
    return Vector4d(rect_center(0), rect_center(1), widthheight(0),
                    widthheight(1));
  }
};

class VertexQuadric : public BaseVertex<9, Quadric>  // NOTE  this vertex stores
                                                     // object pose to world
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  VertexQuadric(){};

  virtual void setToOriginImpl() { _estimate = Quadric(); }

  virtual void oplusImpl(const double* update_) {
    Eigen::Map<const Vector9d> update(update_);
    setEstimate(_estimate.exp_update(update));
  }

  virtual bool read(std::istream& is) {
    Vector9d est;
    for (int i = 0; i < 9; i++)
      is >> est[i];
    Quadric oneQuadric;
    oneQuadric.fromMinimalVector(est);
    setEstimate(oneQuadric);
    return true;
  }

  virtual bool write(std::ostream& os) const {
    Vector9d lv = _estimate.toMinimalVector();
    for (int i = 0; i < lv.rows(); i++) {
      os << lv[i] << " ";
    }
    return os.good();
  }
};

// camera -object 2D projection error, rectangle difference, could also change
// to iou
class EdgeSE3QuadricProj
    : public BaseBinaryEdge<4, Vector4d, VertexSE3Expmap, VertexQuadric> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeSE3QuadricProj(){};

  virtual bool read(std::istream& is) { return true; };

  virtual bool write(std::ostream& os) const { return os.good(); };

  void computeError() {
    const VertexSE3Expmap* SE3Vertex = static_cast<const VertexSE3Expmap*>(
        _vertices[0]);  //  world to camera pose
    const VertexQuadric* quadricVertex = static_cast<const VertexQuadric*>(
        _vertices[1]);  //  object pose to world

    SE3Quat cam_pose_Tcw = SE3Vertex->estimate();
    Quadric global_quadric = quadricVertex->estimate();

    Vector4d rect_project = global_quadric.projectOntoImageBbox(
        cam_pose_Tcw, Kalib);  // center, width, height

    _error = rect_project - _measurement;
  }

  Matrix3d Kalib;
};
}  // namespace g2o
