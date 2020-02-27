#pragma once

#include <quadric_slam/g2o_Object.h>
#include <Eigen/Dense>
#include <build_quadric.hpp>
#include <vector>

#include "distribution.hpp"

constexpr double QUALITY_THRESHOLD = 0.5;
constexpr int TOTALL_CLASS = 10;  // class of detection

class Detection_result {
 public:
  Vector4d bbox;
  double prop;
  int total_id;
  int frame_seq_id;

  Detection_result(Vector5d raw_2d_objs, int input_frame_seq_id)
      : bbox(raw_2d_objs.head(4)),
        prop(raw_2d_objs(4)),
        frame_seq_id(input_frame_seq_id) {}
};

enum DETECT_RESULT { NO_QUADRIC, NEW_QUADRIC, UPDATE_QUADRIC };

class Quadric_landmark {
 public:
  g2o::Quadric Quadric_meas;  // cube_value
  std::shared_ptr<g2o::VertexQuadric> quadric_vertex;
  double meas_quality = 0.6;  // [0,1] the higher, the better
  std::vector<std::shared_ptr<Detection_result>> quadric_tracking;
  int class_id;
  int totall_id;
  DETECT_RESULT isDetected;  //
  int landmark_id;
  double classPro;
  ds::CatDS ds;

  Quadric_landmark(int totall_id)
      : class_id(-1),
        totall_id(totall_id),
        isDetected(NO_QUADRIC),
        classPro(0),
        ds(ds::CatDS(TOTALL_CLASS)) {}
  Quadric_landmark(const Quadric_landmark&) = delete;
  Quadric_landmark& operator=(const Quadric_landmark&) = delete;

  void quadric_detection(
      const Eigen::Matrix<double, 3, 3>& calib,
      vector<Eigen::Matrix<double, 3, 4>,
             Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>>
          projection_matrix) {  //        // Todo:detect or update quadric
                                // 0 no quadric, 1 new quadric, 2 update quadric
    if (isDetected == NEW_QUADRIC || isDetected == UPDATE_QUADRIC) {
      cout << "quadric has been detected" << endl;
      isDetected = UPDATE_QUADRIC;
      // update class of landmark
      ds.update(quadric_tracking.back()->total_id);
      ds.maxPro(class_id, classPro);
      return;
    }
    if (quadric_tracking.size() < 20) {
      cout << "need more frame, size: " << quadric_tracking.size() << endl;
      isDetected = NO_QUADRIC;
      return;
    }

    // compute quadric
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        bboxes;
    for (auto bbox = quadric_tracking.begin(); bbox != quadric_tracking.end();
         ++bbox) {
      bboxes.push_back((*bbox)->bbox.head(4));
    }
    assert(projection_matrix.size() == bboxes.size());

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        lines;
    ComputeLineMat(bboxes, lines);
    //    std::cout << "lines.size: " << lines.size() << std::endl;
    assert(lines.size() / 4 == projection_matrix.size());

    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        planes;
    ComputePlanesMat(calib, projection_matrix, lines, planes);

    std::vector<Eigen::Matrix<double, 1, 10>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 1, 10>>>
        planes_parameter;

    ComputePlanesParameters(planes, planes_parameter);

    Eigen::Matrix3d rotation;
    Eigen::Vector3d shape;
    Eigen::Vector3d translation;
    Eigen::Matrix4d constrained_quadric;

    ComputeDualQuadric(planes_parameter, rotation, shape, translation,
                       constrained_quadric);

    if (meas_quality > QUALITY_THRESHOLD) {
      isDetected = NEW_QUADRIC;
      //      cout << "rotation.det" << rotation.determinant() << endl;
      Quadric_meas = g2o::Quadric(rotation, translation, shape);
      //      for (auto matrix = projection_matrix.begin();
      //           matrix != projection_matrix.end(); ++matrix) {
      //        cout << distance(projection_matrix.begin(), matrix) << endl;
      //        cout << "project : " << calib * (*matrix) << endl;
      //        cout << "project back: "
      //             << Quadric_meas.projectOntoImageBbox(
      //                    g2o::SE3Quat(matrix->block(0, 0, 3, 3),
      //                                 matrix->block(0, 3, 3, 1)),
      //                    calib)
      //             << endl;
      //      }
      //      Quadric_meas.fromMinimalVector(minimalVector);

      //      cout << "minimalVector" << minimalVector << endl;
      //      cout << "Quadric_meas.pose " << Quadric_meas.pose << endl;
      //      cout << "Quadric_meas.scale " << Quadric_meas.scale << endl;
      quadric_vertex = std::make_shared<g2o::VertexQuadric>();
      quadric_vertex->setEstimate(Quadric_meas);

      //      cout << "detection  result: NEW_QUADRIC" << minimalVector << endl;
      return;
    } else {
      //      cout << "detection  result: NO_QUADRIC" << minimalVector << endl;
      isDetected = NO_QUADRIC;
      return;
    }
  }
};
