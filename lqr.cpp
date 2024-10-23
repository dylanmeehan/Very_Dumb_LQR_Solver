#include "lqr.h"


bool LqrSolution::Solve(Eigen::Vector2d &K,
                        Eigen::Matrix2d &S,
                        const Eigen::Matrix2d &A,
                        const Eigen::Vector2d &B,
                        const Eigen::Matrix2d &Q,
                        float R){

    K(0) = 1.0;
    K(1) = 1.0;
    
    S(0,0) = 1.0;
    S(0,1) = 1.0;
    S(1,0) = S(0,1);
    S(1,1) = 1.0;

    return true;
}