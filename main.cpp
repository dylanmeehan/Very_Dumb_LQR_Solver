#include <iostream>
#include "lqr.h"
#include <Eigen/Dense>

int main(){

    const int x_dim = 3;
    const int u_dim = 1;

    Eigen::Matrix<float, x_dim, x_dim> A {{0.0,1.0, 0.0},
                                        {0.0,0.0,1.0},
                                        {0.0, 0.0, 0.0}};
    Eigen::Matrix<float, x_dim, u_dim> B {0.0, 0.0, 1.0};
    Eigen::Matrix<float, x_dim, x_dim> Q {{1.0,0.0, 0.0},
                                         {0.0,1.0, 0.0},
                                         {0.0, 0.0, 1.0}};
    Eigen::Matrix<float, u_dim, u_dim> R {1.0};

    Eigen::Matrix<float, u_dim, x_dim> K;
    Eigen::Matrix<float, x_dim, x_dim> S; 
    Eigen::Matrix<float, x_dim, x_dim> Riccati;

    float delta = 1e-3;
    float epsilon = 1e-6;

    std::cout << " solving ... " << std::endl;

    bool success = LqrSolution::Solve<float, x_dim, u_dim>(K,S,Riccati,A,B,Q,R, 
                                                                delta, epsilon);


    std::cout << "Riccati = " <<std::endl << Riccati << std::endl << std::endl; 
    std::cout << "S = " << std::endl << S << std::endl;

    Eigen::Vector<float, x_dim> eigenvalues = S.eigenvalues().real();
    std::cout << "Eigenvalues of S" << std::endl;
    std::cout << eigenvalues << std::endl << std::endl;
    std::cout << "K = " << std::endl << K << std::endl;


    return 0;
}