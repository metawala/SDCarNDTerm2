#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Predict the state
  x_          = F_ * x_;
  MatrixXd fT = F_.transpose();
  P_          = F_ * P_ * fT + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Update the state by using Kalman Filter equations
  VectorXd zPred = H_ * x_;
  VectorXd y     = z - zPred;
  MatrixXd HT    = H_.transpose();
  MatrixXd S     = H_ * P_ * HT + R_;
  MatrixXd SI    = S.inverse();
  MatrixXd PHT   = P_ * HT;
  MatrixXd K     = PHT * SI;

  // New Estimate
  x_         = x_ + (K * y);
  long xSize = x_.size();
  MatrixXd I = MatrixXd::Identity(xSize, xSize);
  P_         = (I - K*H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Update the state by using Extended Kalman Filter equations
  // Extract parameters
  double px     = x_(0);
  double py     = x_(1);
  double vx     = x_(2);
  double vy     = x_(3);
  double rho    = sqrt(px*px + py*py);
  double theta  = atan2(py, px);
  double rhoDot = (px*vx + py*vy)/rho;

  // Calculate h transformation matrix and y for EKF
  VectorXd h = VectorXd(3);
  h << rho, theta, rhoDot;
  VectorXd y = z - h;

  // Normalizing the angles
  while(y(1)>M_PI || y(1)<-M_PI){
    if(y(1)>M_PI){
      y(1) -= M_PI;
    }
    else{
      y(1) += M_PI;
    }
  }

  // Update with y using standard equations
  MatrixXd HT = H_.transpose();
  MatrixXd S  = H_ * P_ * HT + R_;
  MatrixXd SI = S.inverse();
  MatrixXd K  = P_ * HT * SI;

  // New Estimate
  x_         = x_ + (K * y);
  long xSize = x_.size();
  MatrixXd I = MatrixXd::Identity(xSize, xSize);
  P_         = (I - K*H_) * P_;
}
