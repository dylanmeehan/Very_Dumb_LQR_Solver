#include <iostream>
#include "lqr.h"
#include <Eigen/Dense>

int main(){

    std::cout << "Hi" << std::endl;

    Eigen::Vector2d K;
    Eigen::Matrix2d S;
    Eigen::Matrix2d A {{0.0,1.0},
                       {0.0,0.0}};
    Eigen::Vector2d B {0.0, 1.0};
    Eigen::Matrix2d Q {{1.0,0.0},{0.0,1.0}};
    float R = 1.0;

    bool success = LqrSolution::Solve(K,S,A,B,Q,R);

    std::cout << "S = " << std::endl << S << std::endl;
    std::cout << "K = " << std::endl << K << std::endl;


    return 0;
}