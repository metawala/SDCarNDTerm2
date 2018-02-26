#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Initialize RMSE vector
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // Check the validity of the following inputs:
  //    *the estimations vector should not be zero
  //    *the estimations vector size should be equal to the ground_truth vector size
  if((estimations.size() != ground_truth.size()) || estimations.size()==0){
    std::cout << "Invalid estimations or ground_truth data" << std::endl;
    return rmse;
  }

  // Accumulate residuals
  for(unsigned int i=0; i<estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // Calculate the mean
  rmse = rmse/estimations.size();

  // Calculate the square root
  rmse = rmse.array().sqrt();

  // Return Calculated RMSE
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  // Define a Jacobian Matrix
  MatrixXd Hj(3, 4);

  // Check if input has the same number of columns as Hj
  if(x_state.size() != 4){
    std::cout << "CalculateJacobian() - Error - The state vector must be of size 4" << std::endl;
    return Hj;
  }

  // Recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  // Pre computed set of terms
  double c1 = px*px + py*py;
  double c2 = sqrt(c1);
  double c3 = (c1*c2);

  // Check division by zero
  if(fabs(c1) < 0.001){
    std::cout << "CalculateJacobian() - Error - Divide by zero." << std::endl;
    return Hj;
  }

  // Calculate the Jacobian
  Hj << (px/c2), (py/c2), 0, 0,
    -(py/c1), (px/c1), 0, 0,
    py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  // Return Calculated Jacobian
  return Hj;
}
