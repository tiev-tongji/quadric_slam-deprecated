#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace ds {
enum DIRECTION { INCREASE, DECREASE };
class DrProcess {
 public:
  int totallClass;
  Eigen::VectorXd params;
  int sum;
  double alpha;  // para for  new class;

  DrProcess(double alpha) : totallClass(0), alpha(alpha), sum(0) {}

  double calProb(int classNum) {
    if (classNum > totallClass)
      return 0.0;
    else if (classNum == totallClass)
      return alpha / (sum + alpha);
    else
      return params(classNum) / (sum + alpha);
  }
  bool update(int classNum, DIRECTION drc) {
    if (classNum >= totallClass) {
      return false;
    }
    if (drc == INCREASE) {
      params(classNum)++;
      sum++;
      return true;
    } else {
      params(classNum)--;
      sum--;
      if (params(classNum) < 0)
        params(classNum) = 0;
      if (sum < 0)
        sum = 0;
      return true;
    }
  }
  bool newClass() {
    totallClass++;
    params.resize(totallClass);
    params(totallClass - 1) = 1.0;
    sum++;
  }
};
class DirDS {
 private:
  int totallClass;  // num of categories;
  Eigen::VectorXd params;
  Eigen::VectorXd samples;
  std::gamma_distribution<double> gamma;
  std::default_random_engine generator;

 public:
  explicit DirDS(int totallClass)
      : totallClass(totallClass),
        params(Eigen::VectorXd::Zero(totallClass)),
        samples(Eigen::VectorXd::Zero(totallClass)),
        gamma(1.0, 1.0),
        generator() {}

  bool update(int classNum) {
    if (classNum >= totallClass) {
      std::cerr << "more than totallClass\n";
      return false;
    } else {
      ++params(classNum);
      return true;
    }
  }

  Eigen::VectorXd sampling() {
    samples.resize(totallClass);
    for (int i = 0; i < totallClass; i++) {
      gamma.param(std::gamma_distribution<double>::param_type(params(i), 1.0));
      samples(i) = gamma(generator);
    }
    double sum = samples.sum();
    samples = samples / sum;
    return samples;
  }
};

class CatDS {
 private:
  int totallClass;
  Eigen::VectorXd params;
  DirDS dir;

 public:
  CatDS(int totallClass)
      : totallClass(totallClass),
        params(Eigen::VectorXd::Zero(totallClass)),
        dir(totallClass) {}

  bool update(int classNum) {
    dir.update(classNum);
    params = dir.sampling();
    return true;
  }
  double calProb(int classNum) {
    if (classNum >= totallClass)
      return 0.0;
    else {
      return params(classNum);
    }
  }
  void maxPro(int& classNum, double& pro) {
    Eigen::VectorXd::Index maxClass;
    pro = params.maxCoeff(&maxClass);
    classNum = int(maxClass);
    return;
  }
};
};  // namespace ds

#endif  // DISTRIBUTION_HPP
