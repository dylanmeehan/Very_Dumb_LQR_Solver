#include <Eigen/Dense>

class LqrSolution{
  public:
    static bool Solve(Eigen::Vector2d &K,
                        Eigen::Matrix2d &S,
                        const Eigen::Matrix2d &A,
                        const Eigen::Vector2d &B,
                        const Eigen::Matrix2d &Q,
                        float R);
};