#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/StdVector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <set>
#include <sstream>
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

#define Pi 3.1415926
#define MAX_DETECTIONS 10

struct CornerPoints {
  double left_x, left_y, right_x, right_y;
  CornerPoints() {}
  CornerPoints(double l_x, double l_y, double r_x, double r_y)
      : left_x(l_x), left_y(l_y), right_x(r_x), right_y(r_y) {}
};

void RecordDetectedObjects(string filename,
                           multimap<int, CornerPoints>& objects_info) {
  ifstream file;
  file.open(filename.data());
  if (!file.is_open()) {
    cout << "Unable to Open the Detection File" << endl;
    exit(1);
  }
  vector<string> vec;
  string temp;
  while (getline(file, temp)) {
    vec.push_back(temp);
  }
  cout << "the Raw Detection Date is" << endl;
  for (vector<string>::iterator it = vec.begin(); it != vec.end(); it++) {
    cout << *it << endl;
    istringstream is(*it);
    string s0, s1, s2, s3, s4, s5, s6;
    is >> s0 >> s1 >> s2 >> s3 >> s4 >> s5 >> s6;
    double left_x = atof(s1.c_str());
    double left_y = atof(s2.c_str());
    double right_x = atof(s3.c_str());
    double right_y = atof(s4.c_str());
    int id = atof(s5.c_str());
    objects_info.insert(pair<int, CornerPoints>(
        id, CornerPoints(left_x, left_y, right_x, right_y)));
  }
  cout << endl;
  cout << "the Detected Objects Information is" << endl;

  for (multimap<int, CornerPoints>::iterator it = objects_info.begin();
       it != objects_info.end(); it++) {
    cout << it->first << " " << it->second.left_x << " " << it->second.left_y
         << " " << it->second.right_x << " " << it->second.right_y << " "
         << objects_info.count(it->first) << endl;
  }
  cout << endl;
}

void DrawBoundingBox() {
  Mat img = imread("/home/csj/Documents/quadric/dataset/dish/1.png",
                   CV_LOAD_IMAGE_COLOR);
  rectangle(img, Point(263, 252), Point(341, 327), Scalar(0, 0, 255), 1, 1, 0);
  rectangle(img, Point(430, 213), Point(535, 278), Scalar(255, 0, 0), 1, 1, 0);
  rectangle(img, Point(60, 236), Point(160, 291), Scalar(0, 255, 0), 1, 1, 0);
  rectangle(img, Point(261, 229), Point(361, 256), Scalar(0, 255, 255), 1, 1,
            0);
  imshow("img", img);
  waitKey(0);
  imwrite("/home/csj/Documents/quadric/dataset/boundingbox/1.png", img);
}

void ComputePointMat(multimap<int, CornerPoints> objects_info,
                     int detected_id,
                     int detected_nums,
                     std::vector<Point2d>& points) {
  int num = 0;
  multimap<int, CornerPoints>::iterator it = objects_info.find(detected_id);
  for (multimap<int, CornerPoints>::iterator p = it;
       it != objects_info.end() && num < detected_nums; p++, num++) {
    Point2d top_left(p->second.left_x, p->second.left_y);
    Point2d botton_right(p->second.right_x, p->second.right_y);
    // cout<<top_left<<" "<<botton_right<<endl;
    points.push_back(top_left);
    points.push_back(botton_right);
  }
}

// void
// ComputeLineMat(std::vector<Point2d>points,std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>
// >& lines){
//  for(int i=0;i<points.size();i+=2){
//    lines.push_back(Eigen::Vector3d(1,0,-points[i].x));
//    lines.push_back(Eigen::Vector3d(0,1,-points[i].y));
//    lines.push_back(Eigen::Vector3d(1,0,-points[i+1].x));
//    lines.push_back(Eigen::Vector3d(0,1,-points[i+1].y));
//  }
//}
void ComputeLineMat(
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        bboxes,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>&
        lines) {
  for (int i = 0; i < bboxes.size(); i += 2) {
    lines.push_back(Eigen::Vector3d(1, 0, -bboxes[i](0) - bboxes[i](2) / 2));
    lines.push_back(Eigen::Vector3d(0, 1, -bboxes[i](1) - bboxes[i](3) / 2));
    lines.push_back(Eigen::Vector3d(1, 0, -bboxes[i](0) + bboxes[i](2) / 2));
    lines.push_back(Eigen::Vector3d(0, 1, -bboxes[i](1) + bboxes[i](3) / 2));
  }
}
Eigen::Matrix3d Quater2Rotation(const double& x,
                                const double& y,
                                const double& z,
                                const double& w) {
  Eigen::Quaterniond q;
  q.x() = x;
  q.y() = y;
  q.z() = z;
  q.w() = w;

  Eigen::Matrix3d R = q.normalized().toRotationMatrix();
  return R;
}

void ComputeProjectionMat(
    string filename,
    std::vector<Eigen::Matrix<double, 3, 4>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>>&
        projection_matrix) {
  ifstream file;
  file.open(filename.data());
  if (!file.is_open()) {
    cout << "Unable to Open the Trajectories File" << endl;
    exit(1);
  }
  vector<string> vec;
  string temp;
  while (getline(file, temp)) {
    vec.push_back(temp);
  }

  Eigen::Matrix3d k;
  k << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;

  ofstream f_rotation("camera_rt35.txt");
  // ofstream f_rotation_mat("camera_test.txt");
  for (vector<string>::iterator it = vec.begin(); it != vec.end(); it++) {
    istringstream is(*it);
    string s0, s1, s2, s3, s4, s5, s6;
    is >> s0 >> s1 >> s2 >> s3 >> s4 >> s5 >> s6;
    double tx = atof(s0.c_str());
    double ty = atof(s1.c_str());
    double tz = atof(s2.c_str());
    double qx = atof(s3.c_str());
    double qy = atof(s4.c_str());
    double qz = atof(s5.c_str());
    double qw = atof(s6.c_str());

    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;

    Eigen::Matrix4d cam_rt;
    cam_rt = Eigen::Matrix4d::Identity();
    rotation = Quater2Rotation(qx, qy, qz, qw);

    cam_rt.block(0, 0, 3, 3) = rotation;
    cam_rt.block(0, 3, 3, 1) = Eigen::Vector3d(tx, ty, tz);
    f_rotation << cam_rt.col(0).transpose() << " " << cam_rt.col(1).transpose()
               << " " << cam_rt.col(2).transpose() << " "
               << cam_rt.col(3).transpose() << endl;

    translation << -tx, -ty, -tz;

    Eigen::Matrix<double, 3, 4> projection, rt;

    rt.block(0, 0, 3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1;

    rt.block(0, 0, 3, 3) = rotation.inverse();
    rt.block(0, 3, 3, 1) = rotation.inverse() * translation;  //?

    projection = k * rt;

    projection_matrix.push_back(projection);
  }
  for (int i = 0; i < projection_matrix.size(); i++) {
    cout << "The Projection of Frame " << i << " is " << endl
         << projection_matrix[i] << endl
         << endl;
  }
}

void ComputePlanesMat(
    const vector<Eigen::Matrix<double, 3, 4>,
                 Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>>&
        projection_matrix,
    const vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>&
        lines,
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>&
        planes) {
  for (int i = 0; i < lines.size(); i++) {
    planes.push_back(projection_matrix[i / 4].transpose() * lines[i]);
  }
}

void ComputePlanesParameters(
    vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& planes,
    std::vector<Eigen::Matrix<double, 1, 10>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 1, 10>>>&
        planes_parameter) {
  for (vector<Eigen::Vector4d,
              Eigen::aligned_allocator<Eigen::Vector4d>>::iterator it =
           planes.begin();
       it != planes.end(); it++) {
    Eigen::Matrix<double, 1, 10> parm;

    parm << pow((*it)(0, 0), 2), 2 * (*it)(0, 0) * (*it)(1, 0),
        2 * (*it)(0, 0) * (*it)(2, 0), 2 * (*it)(0, 0) * (*it)(3, 0),
        pow((*it)(1, 0), 2), 2 * (*it)(1, 0) * (*it)(2, 0),
        2 * (*it)(1, 0) * (*it)(3, 0), pow((*it)(2, 0), 2),
        2 * (*it)(2, 0) * (*it)(3, 0), pow((*it)(3, 0), 2);

    planes_parameter.push_back(parm);
  }
}

void ComputeDualQuadric(
    const vector<Eigen::Matrix<double, 1, 10>,
                 Eigen::aligned_allocator<Eigen::Matrix<double, 1, 10>>>&
        planes_parameter,
    Eigen::Vector3d& rotation,
    Eigen::Vector3d& shape,
    Eigen::Vector3d& translation,
    Eigen::Matrix4d& constrained_quadric) {
  ofstream ofile("quadric_parameters.txt", ios::app);

  int rows = planes_parameter.size();
  cout << "rows=" << rows << endl;
  Eigen::MatrixXd planes_svd(rows, 10);
  for (int i = 0; i < rows; i++) {
    planes_svd.row(i) = Eigen::VectorXd::Map(&planes_parameter[i][0],
                                             planes_parameter[i].size());
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      planes_svd, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 1, 10> dual_quadric_parm = svd.matrixV().col(9);

  Eigen::Matrix4d dual_quadric, raw_quadric;

  dual_quadric << dual_quadric_parm(0, 0), dual_quadric_parm(0, 1),
      dual_quadric_parm(0, 2), dual_quadric_parm(0, 3), dual_quadric_parm(0, 1),
      dual_quadric_parm(0, 4), dual_quadric_parm(0, 5), dual_quadric_parm(0, 6),
      dual_quadric_parm(0, 2), dual_quadric_parm(0, 5), dual_quadric_parm(0, 7),
      dual_quadric_parm(0, 8), dual_quadric_parm(0, 3), dual_quadric_parm(0, 6),
      dual_quadric_parm(0, 8), dual_quadric_parm(0, 9);

  raw_quadric = dual_quadric.inverse() * cbrt(dual_quadric.determinant());
  cout << "The Primary Form of Quadric is " << endl
       << raw_quadric << endl
       << endl;

  Eigen::Matrix3d quadric_33;
  quadric_33 = raw_quadric.block(0, 0, 3, 3);
  double det = raw_quadric.determinant() / quadric_33.determinant();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(quadric_33);

  Eigen::Matrix3d eigen_vectors = eigen_solver.eigenvectors();
  cout << "The Rotation of the Constrained Quadric is " << endl
       << eigen_vectors << endl
       << endl;

  Eigen::Vector3d eigen_values;
  eigen_values = eigen_solver.eigenvalues();

  Eigen::Vector3d eigen_values_inverse;
  eigen_values_inverse = eigen_values.array().inverse();

  cout << "eigen_values  " << eigen_values << endl;
  cout << "eigen_values_inverse  " << eigen_values_inverse << endl;

  shape << (((-det) * eigen_values_inverse).array().abs()).array().sqrt();
  cout << "The Shape of the Constrained Quadric is " << endl
       << shape << endl
       << endl;

  translation << dual_quadric_parm(0, 3) / dual_quadric_parm(0, 9),
      dual_quadric_parm(0, 6) / dual_quadric_parm(0, 9),
      dual_quadric_parm(0, 8) / dual_quadric_parm(0, 9);

  cout << "The Translation of the Constrained Quadric is " << endl
       << translation << endl
       << endl;
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

  // T.block(0,0,3,1)=eigen_vectors*translation;
  T.block(0, 0, 3, 3) = eigen_vectors;
  T.block(0, 3, 3, 1) = translation;

  ofile << shape.transpose() << " " << T.col(0).transpose() << " "
        << T.col(1).transpose() << " " << T.col(2).transpose() << " "
        << T.col(3).transpose() << endl;
  Eigen::Vector4d shape_parm;
  shape_parm << pow(shape(0, 0), 2), pow(shape(1, 0), 2), pow(shape(2, 0), 2),
      -1;

  Eigen::Matrix4d quadric_shape(shape_parm.asDiagonal());

  constrained_quadric = T * quadric_shape * T.transpose();
  cout << "The Constrained Quadric Matrix is " << endl
       << constrained_quadric << endl
       << endl;
}

void ComputeConicsMat(
    const Eigen::Matrix4d& constrained_quadric,
    const vector<Eigen::Matrix<double, 3, 4>,
                 Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>>&
        projection_matrix,
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>&
        conics_matrix) {
  Eigen::Matrix3d dual_conic;
  for (int i = 0; i < projection_matrix.size(); i++) {
    dual_conic = projection_matrix[i] * constrained_quadric *
                 projection_matrix[i].transpose();
    conics_matrix.push_back(dual_conic.inverse() *
                            cbrt(dual_conic.determinant()));
  }
  for (int i = 0; i < projection_matrix.size(); i++) {
    cout << "The conic of frame " << i << " is " << endl
         << conics_matrix[i] << endl
         << endl;
  }
}
