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
typedef Eigen::Matrix<double, 4, 1> Vector4d;
namespace g2o {

class Quadric {
 public:
  SE3Quat pose;
  Vector3d scale;  // semi-axis a,b,c
  Quadric() {
    pose = SE3Quat();
    scale.setZero();
  }

  Quadric(const Matrix3d& R, const Vector3d& t, const Vector3d& inputScale) {
    std::cout << "R" << R << std::endl;
    pose = SE3Quat(R, t);
    std::cout << "POSE" << pose << std::endl;
    scale = inputScale;
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
    res.pose = SE3Quat::exp(update.head<6>()) * this->pose;
    res.scale = this->scale + update.tail<3>();
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
    //    std::cout << "scale " << scale << std::endl;
    //    std::cout << "pose" << pose << std::endl;
    centreAtOrigin = Eigen::Matrix4d::Identity();
    centreAtOrigin(0, 0) = pow(scale(0), 2);
    centreAtOrigin(1, 1) = pow(scale(1), 2);
    centreAtOrigin(2, 2) = pow(scale(2), 2);
    centreAtOrigin(3, 3) = -1;
    //    std::cout << "centreAtOrigin" << centreAtOrigin << std::endl;
    Matrix4d Z;
    Z = pose.to_homogeneous_matrix();
    //    std::cout << "Z " << Z << std::endl;
    res = Z * centreAtOrigin * Z.transpose();
    //    std::cout << "toSymMat" << res << std::endl;
    return res;
  }

  // get rectangles after projection  [topleft, bottomright]

  Matrix3d toConic(const SE3Quat& campose_wc, const Matrix3d& calib) const {
    //    std::cout << "toConic" << std::endl;
    //    std::cout << "campose_wc"
    //    << campose_wc.to_homogeneous_matrix().block(0, 0, 3, 4) << std::endl;
    Eigen::Matrix<double, 3, 4> P =
        calib * campose_wc.to_homogeneous_matrix().block(
                    0, 0, 3, 4);  // Todo:BUG!! maybe campose_cw
    Matrix4d symMat = this->toSymMat();
    //    std::cout << "P" << P << std::endl;
    Matrix3d conic = (P * symMat * P.transpose());
    return conic.inverse();
  }

  Vector4d projectOntoImageRect(const SE3Quat& campose_wc,
                                const Matrix3d& Kalib) const {
    //    std::cout << "projectOntoImageRect" << std::endl;
    Matrix3d conic = this->toConic(campose_wc, Kalib);
    Vector6d c;
    c << conic(0, 0), conic(0, 1) * 2, conic(1, 1), conic(0, 2) * 2,
        conic(1, 2) * 2, conic(2, 2);
    Vector2d y, x;
    y(0) = (4 * c(4) * c(0) - 2 * c(1) * c(3) +
            sqrt(pow(2 * c(1) * c(3) - 4 * c(0) * c(4), 2) -
                 4 * (pow(c(1), 2) - 4 * c(0) * c(2)) *
                     (pow(c(3), 2) - 4 * c(0) * c(5)))) /
           (2 * (pow(c(1), 2) - 4 * c(2) * c(0)));

    y(1) = (4 * c(4) * c(0) - 2 * c(1) * c(3) -
            sqrt(pow(2 * c(1) * c(3) - 4 * c(0) * c(4), 2) -
                 4 * (pow(c(1), 2) - 4 * c(0) * c(2)) *
                     (pow(c(3), 2) - 4 * c(0) * c(5)))) /
           (2 * (pow(c(1), 2) - 4 * c(2) * c(0)));

    x(0) = (4 * c(3) * c(2) - 2 * c(1) * c(4) +
            sqrt(pow(2 * c(1) * c(4) - 4 * c(2) * c(3), 2) -
                 4 * (pow(c(1), 2) - 4 * c(0) * c(2)) *
                     (pow(c(4), 2) - 4 * c(2) * c(5)))) /
           (2 * (pow(c(1), 2) - 4 * c(2) * c(0)));

    x(1) = (4 * c(3) * c(2) - 2 * c(1) * c(4) -
            sqrt(pow(2 * c(1) * c(4) - 4 * c(2) * c(3), 2) -
                 4 * (pow(c(1), 2) - 4 * c(0) * c(2)) *
                     (pow(c(4), 2) - 4 * c(2) * c(5)))) /
           (2 * (pow(c(1), 2) - 4 * c(2) * c(0)));
    Vector2d bottomright;  // x y
    Vector2d topleft;
    bottomright(0) = x.maxCoeff();
    bottomright(1) = y.maxCoeff();
    topleft(0) = x.minCoeff();
    topleft(1) = y.minCoeff();
    // Todo:conic at boundary
    //    std::cout << "projectOntoImageRect end"
    //              << Vector4d(topleft(0), topleft(1), bottomright(0),
    //                          bottomright(1))
    //              << std::endl;
    return Vector4d(topleft(0), topleft(1), bottomright(0), bottomright(1));
  }

  //  Vector4d projectOntoImageRect(const SE3Quat& campose_wc,
  //                                const Matrix3d& calib) const {
  //    std::cout << "projectOntoImageRect" << std::endl;
  //    Matrix3d conic = this->toConic(campose_wc, calib);
  //    assert(conic.determinant() < 0);
  //    Vector6d c;
  //    c << conic(0, 0), conic(0, 1) * 2, conic(1, 1), conic(0, 2) * 2,
  //        conic(1, 2) * 2, conic(2, 2);
  //    std::cout << "c: " << c << std::endl;
  //    double centerX =
  //        (2 * c(2) * c(3) - c(1) * c(4)) / (c(1) * c(1) - 4 * c(0) * c(2));
  //    double centerY =
  //        (2 * c(0) * c(4) - c(1) * c(3)) / (c(1) * c(1) - 4 * c(0) * c(2));

  //    double theta = abs(
  //        atan((c(2) - c(0) - sqrt((c(0) - c(2)) * (c(0) - c(2)) + c(1) *
  //        c(1))) /
  //             c(1)));

  //    Vector2d bottomright;  // x y
  //    Vector2d topleft;
  //    bottomright(0) = centerX + 10;
  //    bottomright(1) = centerY + 10;
  //    topleft(0) = centerX - 10;
  //    topleft(1) = centerY - 10;
  //    // Todo:conic at boundary
  //    std::cout << "projectOntoImageRect end"
  //              << Vector4d(topleft(0), topleft(1), bottomright(0),
  //                          bottomright(1))
  //              << std::endl;
  //    return Vector4d(topleft(0), topleft(1), bottomright(0), bottomright(1));
  //  }

  // get rectangles after projection  [center, width, height]
  Vector4d projectOntoImageBbox(const SE3Quat& campose_wc,
                                const Matrix3d& Kalib) const {
    //    std::cout << "projectOntoImageBbox" << std::endl;
    Vector4d rect_project = projectOntoImageRect(
        campose_wc, Kalib);  // top_left, bottom_right  x1 y1 x2 y2
    Vector2d rect_center =
        (rect_project.tail<2>() + rect_project.head<2>()) / 2;
    Vector2d widthheight = rect_project.tail<2>() - rect_project.head<2>();
    //    std::cout << "projectOntoImageBbox end" << std::endl;
    //    std::cout << Vector4d(rect_center(0), rect_center(1), widthheight(0),
    //                          widthheight(1))
    //              << std::endl;
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
    //    std::cout << "EdgeSE3QuadricProj computeError" << std::endl;
    const VertexSE3Expmap* SE3Vertex = static_cast<const VertexSE3Expmap*>(
        _vertices[0]);  //  world to camera pose
    const VertexQuadric* quadricVertex = static_cast<const VertexQuadric*>(
        _vertices[1]);  //  object pose to world

    SE3Quat cam_pose_Tcw = SE3Vertex->estimate();
    Quadric global_quadric = quadricVertex->estimate();

    Vector4d rect_project = global_quadric.projectOntoImageBbox(
        cam_pose_Tcw, calib);  // center, width, height
    std::cout << "global_quadric.pose: " << global_quadric.pose << std::endl;
    std::cout << "global_quadric.scale " << global_quadric.scale << std::endl;
    //    std::cout << "_measurement: " << _measurement << std::endl;
    std::cout << "rect_project: " << rect_project << std::endl;

    _error = rect_project - _measurement;
    std::cout << "_error: " << _error << std::endl;
    std::abort();
  }

  Matrix3d calib;
};
}  // namespace g2o
