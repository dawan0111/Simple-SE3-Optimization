#include "sophus/se3.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <cmath>
#include <iostream>
#include <random>

#define RANDOM_SEED 1004

using Vector6d = Eigen::Matrix<double, 6, 1>;

Eigen::Vector4d getHomogenous(const Eigen::Vector3d &point) {
  Eigen::Vector4d homogenousPoint = Eigen::Vector4d::Zero();
  homogenousPoint(0) = point(0);
  homogenousPoint(1) = point(1);
  homogenousPoint(2) = point(2);
  homogenousPoint(3) = 1.0;
  return homogenousPoint;
}

double costFunction(const std::vector<Eigen::Vector3d> &measures,
                    const std::vector<Eigen::Vector3d> &landmarks,
                    const Sophus::SE3<double> &T) {
  double cost = 0.0;
  for (int i = 0; i < landmarks.size(); ++i) {
    auto homogenousMeasure = getHomogenous(measures[i]);
    auto homogenousLandmark = getHomogenous(landmarks[i]);
    cost += pow((homogenousMeasure - T * homogenousLandmark).norm(), 2);
  }
  return cost;
}

Vector6d gaussNewton(const std::vector<Eigen::Vector3d> &measures,
                     const std::vector<Eigen::Vector3d> &landmarks,
                     const Sophus::SE3<double> &T) {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Vector6d b = Vector6d::Zero();
  Eigen::Matrix3d informationMatrix = Eigen::Matrix3d::Identity();

  for (int i = 0; i < measures.size(); ++i) {
    auto measure = measures[i];
    auto landmark = landmarks[i];

    Eigen::Matrix<double, 3, 6> jacobian = Eigen::Matrix<double, 3, 6>::Zero();
    Eigen::Vector3d so3 = Eigen::Vector3d::Zero();
    so3 = T.rotationMatrix() * landmark + T.translation();
    auto error = so3 - measure;

    jacobian.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
    jacobian.block(0, 3, 3, 3) = Sophus::SO3d::hat(so3) * -1.0;

    b += error.transpose() * informationMatrix * jacobian;
    H += jacobian.transpose() * informationMatrix * jacobian;
  }

  Vector6d dx = H.colPivHouseholderQr().solve(-b);
  return dx;
}

int32_t main() {
  std::uniform_real_distribution dist(-1.0, 1.0);
  std::mt19937_64 rng1(RANDOM_SEED);

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  Sophus::SE3 T = Sophus::SE3(R, t);

  std::cout << "initialGauss: " << T.matrix() << std::endl;
  std::vector<Eigen::Vector3d> landmarks;
  std::vector<Eigen::Vector3d> measures;

  landmarks.emplace_back(0.9, 1.0, 1.0);
  landmarks.emplace_back(1.0, 2.0, 1.2);
  landmarks.emplace_back(1.1, 3.0, 1.0);
  landmarks.emplace_back(1.2, 3.5, 1.1);
  landmarks.emplace_back(1.5, 3.7, 1.4);

  for (auto &landmark : landmarks) {
    Eigen::Vector3d noise = Eigen::Vector3d::Zero();
    noise(0) = dist(rng1);
    noise(1) = dist(rng1);
    noise(2) = dist(rng1);
    measures.emplace_back(landmark + noise);
  }

  for (int i = 0; i < 10; ++i) {
    std::cout << "#Iteration " << i << "." << std::endl;
    auto cost = costFunction(measures, landmarks, T);
    std::cout << "cost: " << cost << std::endl;

    auto dT = gaussNewton(measures, landmarks, T);
    T = Sophus::SE3d::exp(dT) * T;
    std::cout << "T: " << std::endl;
    std::cout << T.matrix() << std::endl;
  }
  return 0;
}