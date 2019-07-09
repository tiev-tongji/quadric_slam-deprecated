#pragma once

#include <quadric_slam/g2o_Object.h>
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
class Quadric_landmark {
 public:
  g2o::Quadric Quadric_meas;  // cube_value
  g2o::VertexQuadric* quadric_vertex;
  double meas_quality;  // [0,1] the higher, the better
  std::vector<Detection_result*> quadric_tracking;
  bool isDetected;
  int class_id;
  int quadric_detection() {  //        // Todo:detect or update quadric
    // 0 no quadric, 1 new quadric, 2 update quadric
    //      if (quadric_tracking.size() < 3) {
    //        isDetected = false;
    //        return 0;
    //      } else {
    ////        meas_quality = quality;
    //        if (meas_quality > QUALITY_THRESHOLD) {
    //          isDetected = true;
    //          return 1;
    //        } else {
    //          isDetected = 0;
    //          return isDetected;
    //        }
    //      }
    return 0;
  }
};
