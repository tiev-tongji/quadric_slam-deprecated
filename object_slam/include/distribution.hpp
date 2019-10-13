#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <vector>

namespace ds {
enum DIRECTION { INCREASE, DECREASE };
class DPProcess {
 public:
  int totallClass;
  Eigen::VectorXd params;
  int sum;
  double alpha;  // para for  new class;

  DPProcess(double alpha) : totallClass(0), alpha(alpha) {}

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
        sum == 0;
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
 public:
  int totallClass;  // num of categories;
  Eigen::VectorXd params;
  Eigen::VectorXd samples;

  std::default_random_engine* pGenerator;
  std::gamma_distribution<double>* pGamma;
  DirDS(int totallClass) : totallClass(totallClass) {
    params.resize(totallClass);
    pGamma = new std::gamma_distribution<double>(1.0, 1.0);
    pGenerator = new std::default_random_engine;
  }
  bool update(int classNum) {
    if (classNum >= totallClass)
      return false;
    else {
      params(classNum)++;
      return true;
    }
  }
  Eigen::VectorXd sampling() {
    samples.resize(totallClass);
    for (int i = 0; i < totallClass; i++) {
      pGamma->param(params(i), 1.0);
      samples(i) = (*pGamma)(*pGenerator);
    }
    double sum = samples.sum();
    samples = samples / sum;
    return samples;
  }
};

class CatDS {
 public:
  int totallClass;
  Eigen::VectorXd params;
  DirDS* pDir;
  CatDS(int totallClass) : totallClass(totallClass) {
    params.resize(params);
    pDir = new DirDS(totallClass);
  }
  bool update(int classNum) {
    pDir->update(classNum);
    params = pDir->sampling();
  }
  double calProb(int classNum) {
    if (classNum >= totallClass)
      return 0.0;
    else {
      return params(classNum);
    }
  }
  void maxPro(int& classNum, double& pro) {
    pro = params.max(classNum);
    return;
  }
};
};  // namespace ds

#endif  // DISTRIBUTION_HPP
