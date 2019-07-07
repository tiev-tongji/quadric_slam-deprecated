#pragma once

#include <vector>

#include <quadric_slam/g2o_Object.h>
class Quadric_landmark {
 public:
  g2o::Quadric Quadric_meas;  // cube_value
  g2o::VertexQuadric* quadric_vertex;
  double meas_quality;  // [0,1] the higher, the better
};
