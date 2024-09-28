#include <iostream>
#include "Eigen/Dense"
#include "initial_state_array.h"  // Include your array definitions
#include "A_array.h"
#include "W_array.h"
#include "Q_array.h"
#include "H_array.h"
#include "measurements_array.h"
using namespace Eigen;

// Define constants
const int states = 6;
const int neurons = 164;
int total_time_stamps = 2;
// Function to print Eigen vector or matrix
template<typename Derived>
void printMatrix(const Eigen::MatrixBase<Derived>& mat) {
    std::cout << mat << std::endl;
}

// Define the BCIKalmanFilter class in the same file
class BCIKalmanFilter {
public:
    // Constructor
    BCIKalmanFilter() 
        : vec_X_(VectorXf::Zero(states)),  // Initialize vector with zero
          mat_P_(Matrix<float, states, states>::Zero()),  // Initialize P as zero
          mat_F_(Matrix<float, states, states>::Zero()),  // Initialize F as zero
          mat_Q_(Matrix<float, states, states>::Zero()),  // Initialize Q as zero
          mat_R_(Matrix<float, neurons, neurons>::Zero()), // Initialize R as zero
          mat_H_(Matrix<float, neurons, states>::Zero())   // Initialize H as zero
    {}

    // Member functions to get/set vectors and matrices
    VectorXf& Vec_X() { return vec_X_; }
    Matrix<float, states, states>& Mat_P() { return mat_P_; }
    Matrix<float, states, states>& Mat_F() { return mat_F_; }
    Matrix<float, states, states>& Mat_Q() { return mat_Q_; }
    Matrix<float, neurons, neurons>& Mat_R() { return mat_R_; }
    Matrix<float, neurons, states>& Mat_H() { return mat_H_; }

    // Prediction and correction function
    void predict_and_correct(const Vector<float, neurons> &Vec_Z/*measurement vector*/) {
        // Prediction step
        // vec_X_ = mat_F_ * vec_X_;  // State prediction
        // mat_P_ = mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_;  // Covariance prediction
        
        Vector<float, neurons> Vec_Y = Vec_Z - (mat_H_ * mat_F_ * vec_X_);

        Matrix<float, neurons, neurons> Mat_S = mat_H_ * (mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_) * mat_H_.transpose() + mat_R_;
        
        
        Matrix<float, states, neurons> Mat_K = (mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_) * mat_H_.transpose() * Mat_S.inverse();

        Matrix<float, states, states> Mat_I = Matrix<float, states, states>::Identity();

        vec_X_ = mat_F_ * vec_X_  + Mat_K * Vec_Y;
        
        mat_P_ = (Mat_I -  Mat_K * mat_H_) * (mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_);

        // vec_X_ = vec_X_ + K * (Vec_Z - mat_H_ * vec_X_);  // Update state with measurement
        // mat_P_ = (Matrix<float, states, states>::Identity() - K * mat_H_) * mat_P_;  // Update covariance
    }

private:
    // Member variables for vectors and matrices
    VectorXf vec_X_;
    Matrix<float, states, states> mat_P_;
    Matrix<float, states, states> mat_F_;
    Matrix<float, states, states> mat_Q_;
    Matrix<float, neurons, neurons> mat_R_;
    Matrix<float, neurons, states> mat_H_;
};

int main() {
    // Instantiate the bci_kalman_filter object
    BCIKalmanFilter bci_kalman_filter;

    // Mapping the initial state vector
    extern float initial[];  // Declare external float array
    VectorXf vec_X = Map<VectorXf>(initial, states);
    bci_kalman_filter.Vec_X() = vec_X;

    // std::cout << vec_X << std::endl;


    // Zero initialization for matrix P
    bci_kalman_filter.Mat_P() = Matrix<float, states, states>::Zero();
    // std::cout << bci_kalman_filter.Mat_P() << std::endl;

    // Map matrices from flat arrays (row-major)
    extern float A[];  
    Matrix<float, states, states> mat_F = Map<Matrix<float, states, states, RowMajor>>(A);
    bci_kalman_filter.Mat_F() = mat_F;
    // std::cout << bci_kalman_filter.Mat_F() << std::endl;

    extern float W[];  
    Matrix<float, states, states> mat_Q = Map<Matrix<float, states, states, RowMajor>>(W);
    bci_kalman_filter.Mat_Q() = mat_Q;
    // std::cout << bci_kalman_filter.Mat_Q() << std::endl;

    extern float Q[];  
    Matrix<float, neurons, neurons> mat_R = Map<Matrix<float, neurons, neurons, RowMajor>>(Q);
    bci_kalman_filter.Mat_R() = mat_R;
    // std::cout << bci_kalman_filter.Mat_R() << std::endl;

    extern float H[];  
    Matrix<float, neurons, states> mat_H = Map<Matrix<float, neurons, states, RowMajor>>(H);
    bci_kalman_filter.Mat_H() = mat_H;
    // std::cout << bci_kalman_filter.Mat_H() << std::endl;

    // Optionally print the matrices to verify
    // printMatrix(bci_kalman_filter.Vec_X());
    // printMatrix(bci_kalman_filter.Mat_F());
    // printMatrix(bci_kalman_filter.Mat_Q());
    // printMatrix(bci_kalman_filter.Mat_R());
    // printMatrix(bci_kalman_filter.Mat_H());

  for(int j = 1; j < total_time_stamps/*time_stamps*/; j++)
  {
    Vector<float, neurons> vec_Z;
    for(int i = neurons*j; i < neurons*(j+1); i++)
    {
        vec_Z(i-neurons*j) = measurements[i];    
    }
    std::cout << vec_Z << std::endl;
    bci_kalman_filter.predict_and_correct(vec_Z);
    // std::cout << bci_kalman_filter.Vec_X() << std::endl;
    // std::cout << std::endl;

        // printMatrix(vec_Z);
  }
    return 0;
}
