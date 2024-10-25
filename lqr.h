#pragma once

#include <Eigen/Dense>

class LqrSolution{
  public:
    template<typename T, int x_dim, int u_dim>
    static bool  Solve(Eigen::Matrix<T, u_dim, x_dim> &K,
                        Eigen::Matrix<T, x_dim, x_dim> &S,
                        Eigen::Matrix<T, x_dim, x_dim> &Riccati,
                        const Eigen::Matrix<T, x_dim, x_dim> &A,
                        const Eigen::Matrix<T, x_dim, u_dim> &B,
                        const Eigen::Matrix<T, x_dim, x_dim> &Q,
                        const Eigen::Matrix<T, u_dim, u_dim> &R);
    
    template<typename T, int x_dim>
    static T SumOfSquaresOfMatrixEntries(const Eigen::Matrix<T,x_dim,x_dim> &M);

    template<typename T, int x_dim, int u_dim>
    static Eigen::Matrix<T,x_dim,x_dim> Riccati(
                    const Eigen::Matrix<T, x_dim, x_dim> &S,
                    const Eigen::Matrix<T, x_dim, x_dim> &A,
                    const Eigen::Matrix<T, x_dim, u_dim> &B,
                    const Eigen::Matrix<T, u_dim, u_dim> &R_inv,
                    const Eigen::Matrix<T, x_dim, x_dim>& Q){
                        return S * A + A.transpose() * S - S * B * R_inv * B.transpose() * S + Q;
                    }
};


template<typename T, int x_dim>
T LqrSolution::SumOfSquaresOfMatrixEntries(const Eigen::Matrix<T,x_dim,x_dim> &M){
    T abs_sum_square = 0;
    for (int i = 0; i < x_dim; i++){
        for (int j = 0; j < x_dim; j++){
            abs_sum_square += M(i,j)*M(i,j);
        }
    }
    return abs_sum_square;

}

template<typename T, int x_dim, int u_dim>
bool LqrSolution::Solve( Eigen::Matrix<T, u_dim, x_dim> &K,
                        Eigen::Matrix<T, x_dim, x_dim> &S,
                        Eigen::Matrix<T, x_dim, x_dim> &Ricc,
                        const Eigen::Matrix<T, x_dim, x_dim> &A,
                        const Eigen::Matrix<T, x_dim, u_dim> &B,
                        const Eigen::Matrix<T, x_dim, x_dim> &Q,
                        const Eigen::Matrix<T, u_dim, u_dim> &R){

    // initialize everything before loop
    K.setOnes();
    S.setIdentity();
    Eigen::Matrix<T, u_dim, u_dim> R_inv = R.inverse();
    Eigen::Matrix<T, x_dim, x_dim> A_T = A.transpose();
    Eigen::Matrix<T, u_dim, x_dim> B_T = B.transpose();


    float delta = 1e-3;
    float epsilon = 1e-6;
    int max_iterations = 100000;
    bool printout = 0;

    int i = 1;
    int j = 1;
    int last_i_with_change = i;
    int last_j_with_change = j;
    Eigen::Matrix<T,x_dim,x_dim> S_test;
    int iteration_count = 0;

    while (iteration_count < max_iterations){

        bool has_checked_bigger = 0;
        bool S_ij_changed = 0;
        while(true){

            T S_ij_current_point = S(i,j);
            Eigen::Matrix<T,x_dim, x_dim> Riccati_current = Riccati<T,x_dim,u_dim>(S,A,B,R_inv,Q);
            T objective_function_current = SumOfSquaresOfMatrixEntries<T,x_dim>(Riccati_current);       

            S_test = S;
            if (!has_checked_bigger){
                S_test(i,j) = S(i,j) + delta;
            } else{
                S_test(i,j) = S(i,j) - delta;
            }

            Eigen::Matrix<T,x_dim, x_dim> Riccati_test = Riccati<T,x_dim,u_dim>(S_test,A,B,R_inv,Q);
            T objective_function_test = SumOfSquaresOfMatrixEntries<T,x_dim>(Riccati_test);

            if (printout){
                std::cout << "i " << i << ", j " << j;
                std::cout << "   S_ij_current " << S(i,j) << ", Sij_test " << S_test(i,j);
                std::cout << "   obj current: " << objective_function_current << ", obj test: " << objective_function_test;
            }

            // found better solution 
            if (objective_function_test < objective_function_current - epsilon){
                S(i,j) = S_test(i,j);
                if(printout){
                    std::cout << std::endl << "   updating " << "i " << i << ", j "<<j << std::endl;
                }
                S_ij_changed = true;
            } else{
                if (!has_checked_bigger){
                    has_checked_bigger = true;
                    if(printout){
                        std::cout << "   done checking bigger";
                    }
                    
                } else { //we've checked both bigger and smaller
                    if (printout){
                        std::cout << "   done checking both";
                    }
                    
                    break;
                }
            }
            if (printout){std::cout <<std::endl;}
            
        } 

        if (S_ij_changed){
            last_i_with_change = i;
            last_j_with_change = j;
        }

        j = (j + 1) % x_dim;
        if (j == last_j_with_change) {
            i = (i + 1) % x_dim;
            if (i == last_i_with_change){
                if(printout){
                    std::cout << std::endl << " last i with change " << i << ", last j with change " << j << std::endl;
                }
                break;
            }
        }

        iteration_count++;

    }

    Ricc = Riccati<T,x_dim,u_dim>(S,A,B,R_inv,Q);
    K = R_inv*B_T*S;

    return true;
}