#include <iostream>
#include "Eigen/Dense"
#include <iomanip>
typedef float data_type;
//float MSE 3.82387e-12

#include "A_array.h"
#include "H_array.h"
#include "initial_state_array.h"
#include "measurements_array.h"
#include "P_array.h"
#include "prediction_array.h"
#include "Q_array.h"
#include "real_array.h"
#include "W_array.h"
using namespace Eigen;

// Define constants
const int states = 6;
const int neurons = 164;
int total_time_stamps = 100;
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
          mat_P_(Matrix<data_type, states, states>::Zero()),  // Initialize P as zero
          mat_F_(Matrix<data_type, states, states>::Zero()),  // Initialize F as zero
          mat_Q_(Matrix<data_type, states, states>::Zero()),  // Initialize Q as zero
          mat_R_(Matrix<data_type, neurons, neurons>::Zero()), // Initialize R as zero
          mat_H_(Matrix<data_type, neurons, states>::Zero())   // Initialize H as zero
    {}

    // Member functions to get/set vectors and matrices
    VectorXf& Vec_X() { return vec_X_; }
    Matrix<data_type, states, states>& Mat_P() { return mat_P_; }
    Matrix<data_type, states, states>& Mat_F() { return mat_F_; }
    Matrix<data_type, states, states>& Mat_Q() { return mat_Q_; }
    Matrix<data_type, neurons, neurons>& Mat_R() { return mat_R_; }
    Matrix<data_type, neurons, states>& Mat_H() { return mat_H_; }

    // Prediction and correction function
    void predict_and_correct(const Vector<data_type, neurons> &Vec_Z/*measurement vector*/) {
        // Prediction step
        // vec_X_ = mat_F_ * vec_X_;  // State prediction
        // mat_P_ = mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_;  // Covariance prediction
        
        Vector<data_type, neurons> Vec_Y = Vec_Z - (mat_H_ * mat_F_ * vec_X_);
    // std::cout << Vec_Y << std::endl;

        Matrix<data_type, neurons, neurons> Mat_S = mat_H_ * (mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_) * mat_H_.transpose() + mat_R_;
        // std::cout << mat_H_ * (mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_)* mat_H_.transpose() << std::endl;
        // std::cout << Mat_S << std::endl;
        
        
        Matrix<data_type, states, neurons> Mat_K = (mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_) * mat_H_.transpose() * Mat_S.inverse();
        // std::cout << Mat_K << std::endl;

        Matrix<data_type, states, states> Mat_I = Matrix<data_type, states, states>::Identity();
        // std::cout << Mat_I << std::endl;

        vec_X_ = mat_F_ * vec_X_  + Mat_K * Vec_Y;
        // std::cout << vec_X_ << std::endl;

        mat_P_ = (Mat_I -  Mat_K * mat_H_) * (mat_F_ * mat_P_ * mat_F_.transpose() + mat_Q_);
        // std::cout << mat_P_ << std::endl;

        // vec_X_ = vec_X_ + K * (Vec_Z - mat_H_ * vec_X_);  // Update state with measurement
        // mat_P_ = (Matrix<data_type, states, states>::Identity() - K * mat_H_) * mat_P_;  // Update covariance
    }

private:
    // Member variables for vectors and matrices
    VectorXf vec_X_;
    Matrix<data_type, states, states> mat_P_;
    Matrix<data_type, states, states> mat_F_;
    Matrix<data_type, states, states> mat_Q_;
    Matrix<data_type, neurons, neurons> mat_R_;
    Matrix<data_type, neurons, states> mat_H_;
};

int main() {
    // Instantiate the bci_kalman_filter object
    BCIKalmanFilter bci_kalman_filter;

    // Mapping the initial state vector
    extern data_type initial[];  // Declare external data_type array
    VectorXf vec_X = Map<VectorXf>(initial, states);
    bci_kalman_filter.Vec_X() = vec_X;

    // std::cout << vec_X << std::endl;


    // Zero initialization for matrix P
    bci_kalman_filter.Mat_P() = Matrix<data_type, states, states>::Zero();
    // std::cout << bci_kalman_filter.Mat_P() << std::endl;

    // Map matrices from flat arrays (row-major)
    extern data_type A[];  
    Matrix<data_type, states, states> mat_F = Map<Matrix<data_type, states, states, RowMajor>>(A);
    bci_kalman_filter.Mat_F() = mat_F;
    // std::cout << bci_kalman_filter.Mat_F() << std::endl;

    extern data_type W[];  
    Matrix<data_type, states, states> mat_Q = Map<Matrix<data_type, states, states, RowMajor>>(W);
    bci_kalman_filter.Mat_Q() = mat_Q;
    // std::cout << bci_kalman_filter.Mat_Q() << std::endl;

    extern data_type Q[];  
    Matrix<data_type, neurons, neurons> mat_R = Map<Matrix<data_type, neurons, neurons, RowMajor>>(Q);
    bci_kalman_filter.Mat_R() = mat_R;
    // std::cout << bci_kalman_filter.Mat_R() << std::endl;

    extern data_type H[];  
    Matrix<data_type, neurons, states> mat_H = Map<Matrix<data_type, neurons, states, RowMajor>>(H);
    bci_kalman_filter.Mat_H() = mat_H;
    // std::cout << bci_kalman_filter.Mat_H() << std::endl;

    // Optionally print the matrices to verify
    // printMatrix(bci_kalman_filter.Vec_X());
    // printMatrix(bci_kalman_filter.Mat_F());
    // printMatrix(bci_kalman_filter.Mat_Q());
    // printMatrix(bci_kalman_filter.Mat_R());
    // printMatrix(bci_kalman_filter.Mat_H());
  data_type sum_sqr_vec = 0.0;
  Vector<data_type, states> diff_vec;

  for(int j = 1; j < total_time_stamps/*time_stamps*/; j++)
  {
    Vector<data_type, neurons> vec_Z;
    for(int i = neurons*j; i < neurons*(j+1); i++)
    {
        vec_Z(i-neurons*j) = measurements[i];    
    }
    // std::cout << vec_Z << std::endl;
    bci_kalman_filter.predict_and_correct(vec_Z);

    Vector<data_type, states> vec_python;
    Vector<data_type, states> Vec_X_Final;
    Vec_X_Final = bci_kalman_filter.Vec_X();
    std::cout << "Reference: ";
    for(int i = states*j; i < states*(j+1); i++)
    {
        vec_python(i-states*j) = prediction[i];    
        // std::cout << vec_python(i-states*j) << "\t";
        std::cout << std::fixed << std::setprecision(15) << vec_python(i - states * j) << "\t";
        std::cout << std::defaultfloat;
    }
    std::cout << std::endl;

    std::cout << "Prediction: ";
    for(int i = 0; i < states; i++)
    {
        // std::cout << Vec_X_Final[i] << "\t";
        std::cout << std::fixed << std::setprecision(15) << Vec_X_Final[i] << "\t";
        std::cout << std::defaultfloat;
    }
    std::cout << std::endl;

      diff_vec = Vec_X_Final - vec_python;
      data_type abs_diff = (diff_vec.cwiseAbs()).sum();
      data_type sqr_diff = std::pow(abs_diff,2);
      sum_sqr_vec += sqr_diff;

  }

  sum_sqr_vec = sum_sqr_vec/((total_time_stamps-1)*states);

  std::cout << "MSE is = \n" << sum_sqr_vec << std::endl;  
    return 0;
}
