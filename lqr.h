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
                        const Eigen::Matrix<T, u_dim, u_dim> &R,
                        float delta, float epsilon);
    
    template<typename T, int x_dim>
    static T SumOfSquaresOfMatrixEntries(const Eigen::Matrix<T,x_dim,x_dim> &M);

    template<typename T, int x_dim>
    static T PenalizeNegativeEigenvalues(const Eigen::Matrix<T,x_dim,x_dim> &S);

    template<typename T, int x_dim, int u_dim>
    static Eigen::Matrix<T,x_dim,x_dim> Riccati(
                    const Eigen::Matrix<T, x_dim, x_dim> &S,
                    const Eigen::Matrix<T, x_dim, x_dim> &A,
                    const Eigen::Matrix<T, x_dim, u_dim> &B,
                    const Eigen::Matrix<T, u_dim, u_dim> &R_inv,
                    const Eigen::Matrix<T, x_dim, x_dim> &Q){
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
                        const Eigen::Matrix<T, u_dim, u_dim> &R,
                        float delta, float epsilon){

    // precalculate certain values
    Eigen::Matrix<T, u_dim, u_dim> R_inv = R.inverse();
    Eigen::Matrix<T, x_dim, x_dim> A_T = A.transpose();
    Eigen::Matrix<T, u_dim, x_dim> B_T = B.transpose();


    bool printout = 0;

    bool solution_found = 0;
    int initial_guess = 1;
    int max_guesses = 5;

    while (initial_guess < max_guesses) { // loop over multiple initial guesses of S

        int max_iterations_per_loop = 10000;
        int iteration_count = 0;

        S.setIdentity();
        S = S*initial_guess; // try different guesses of S each loop
        
        int i = 0;
        int j = 0;
        int last_i_with_change = i;
        int last_j_with_change = j;
        Eigen::Matrix<T,x_dim,x_dim> S_test;

        std::cout << "trying with S = " << S << std::endl;

        while (iteration_count < max_iterations_per_loop){ // gradient descent for 1 guess

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

                // keep positive semi definite
                if (S_test(i,j) < 0){
                    S_test(i,j) = 0;
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
                
            } // try bigger and smaller S(i,j)

            if (S_ij_changed){
                last_i_with_change = i;
                last_j_with_change = j;
            }

            // if we have looped over all entries in the matrix without changing S then break 
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
            if (iteration_count == max_iterations_per_loop){
                std::cout << "reached max iterations " << std::endl;
            }

        } // gradient descent for 1 guess

        // if any eigenvalues are < 0 or we do not find a zero , note unsuccessful
        Eigen::Vector<float, x_dim> eigenvalues = S.eigenvalues().real();
        Ricc = Riccati<T,x_dim,u_dim>(S,A,B,R_inv,Q);
        T objective = SumOfSquaresOfMatrixEntries<T,x_dim>(Ricc);

        bool successful_guess = 1;
        if (objective > epsilon){
            successful_guess = 0;
        }
        for (int i = 0; i < x_dim; i ++){
            if (eigenvalues(i) < 0){
                successful_guess = 0;
            }
        }
        if (successful_guess){
            break;
        }

        initial_guess ++;
    } // loop over multiple initial guesses of S

    Ricc = Riccati<T,x_dim,u_dim>(S,A,B,R_inv,Q);
    K = R_inv*B_T*S;

    return solution_found;
}