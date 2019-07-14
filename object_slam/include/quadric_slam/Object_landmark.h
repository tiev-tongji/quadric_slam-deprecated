#pragma once

#include <quadric_slam/g2o_Object.h>
#include <build_quadric.hpp>
#include <vector>

#define QUALITY_THRESHOLD 0.5

class Detection_result {
 public:
  Vector4d bbox;
  double prop;
  int class_id;
  int frame_seq_id;

  Detection_result(Vector5d raw_2d_objs, int frame_seq_id) {
    bbox = raw_2d_objs.head(4);
    prop = raw_2d_objs(4);
  }
};

enum DETECT_RESULT { NO_QUADRIC, NEW_QUADRIC, UPDATE_QUADRIC };

class Quadric_landmark {
 public:
  Quadric_landmark() { isDetected = NO_QUADRIC; }
  g2o::Quadric Quadric_meas;  // cube_value
  g2o::VertexQuadric* quadric_vertex;
  double meas_quality = 1;  // [0,1] the higher, the better
  std::vector<Detection_result*> quadric_tracking;
  DETECT_RESULT isDetected;  //
  int class_id;
  void quadric_detection(
      vector<Eigen::Matrix<double, 3, 4>,
             Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>>
          projection_matrix) {  //        // Todo:detect or update quadric
                                // 0 no quadric, 1 new quadric, 2 update quadric
    if (isDetected == NEW_QUADRIC || isDetected == UPDATE_QUADRIC) {
      isDetected = UPDATE_QUADRIC;
      return;
    }
    if (quadric_tracking.size() < 3) {
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

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        lines;
    ComputeLineMat(bboxes, lines);
    assert(lines.size() / 4 == projection_matrix.size());

    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
        planes;
    ComputePlanesMat(projection_matrix, lines, planes);

    std::vector<Eigen::Matrix<double, 1, 10>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 1, 10>>>
        planes_parameter;

    ComputePlanesParameters(planes, planes_parameter);

    Eigen::Vector3d rotation;
    Eigen::Vector3d shape;
    Eigen::Vector3d translation;
    Eigen::Matrix4d constrained_quadric;

    ComputeDualQuadric(planes_parameter, rotation, shape, translation,
                       constrained_quadric);

    Vector9d minimalVector;
    minimalVector << translation, rotation, shape;
    Quadric_meas.fromMinimalVector(minimalVector);
    quadric_vertex->setEstimate(Quadric_meas);

    if (meas_quality > QUALITY_THRESHOLD) {
      isDetected = NEW_QUADRIC;
      return;
    } else {
      isDetected = NO_QUADRIC;
      return;
    }
  }
};
