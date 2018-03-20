#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  //Calculate the RMSE here.
  VectorXd rmse = VectorXd(4);
  rmse << 0, 0, 0, 0;

  // Error checking for estimations and ground_truth
  if((estimations.size() != ground_truth.size()) || estimations.size() == 0){
    std::cout << "CalculateRMSE() - Error - size error for estimations or ground truth" << std::endl;
    return rmse;
  }

  // Accumulate sqaured residuals
  for(unsigned int i=0; i<estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // Calculate the mean
  rmse = rmse/estimations.size();

  // Calculate the square root
  rmse = rmse.array().sqrt();

  // Return RMSE
  return rmse;
}
