#ifndef idsteach_css_niwmmml
#define idsteach_css_niwmmml

#include <vector>
#include <armadillo>

#include "utils.hpp"
#include "debug.hpp"

double calc_log_z(arma::mat &lambda, 
                  double kappa,
                  double nu)
{
    double d = double(lambda.n_rows);
    double log_z = 0;

    log_z = log(2)*(nu*d/2) + (d/2.0)*log(2*M_PI/kappa) + mvgammaln(nu/2, d) - (nu/2)*logdet(lambda);
    return log_z;
}

double lgniwml(arma::mat &X_k, 
               arma::rowvec &mu_0,
               arma::mat &lambda_0,
               double kappa_0,
               double nu_0,
               double Z_0)
{

    double kappa_n, nu_n;
    arma::mat lambda_n, xbar, S;

    auto n = static_cast<double>(X_k.n_rows); // number of data points
    auto d = X_k.n_cols; // dimensions

    // we have to take precautions because these functions perform
    // differently on row vectors, e.g., when there is only one
    // data point.
    if(n==1){
        xbar = X_k;
        S = arma::zeros(d, d);
    }else{
        S = arma::cov(X_k)*(n-1);
        xbar = arma::mean(X_k);
    }
    
    // Update params
    kappa_n = kappa_0 + n;
    nu_n = nu_0 + n;
    lambda_n = lambda_0 + S + ((kappa_0*n)/(kappa_n))*trans((xbar-mu_0))*(xbar-mu_0);

    double Z_n = calc_log_z(lambda_n, kappa_n, nu_n);

    return Z_n - Z_0 - log(2*M_PI)*(n*d/2);
} 

double lgniwmmml(std::vector<std::vector<double>> X_,
                 std::vector<std::vector<double>> lambda_0_,
                 std::vector<double> mu_0_,
                 double kappa_0,
                 double nu_0,
                 double crp_alpha,
                 std::vector<size_t> Z_start,
                 std::vector<size_t> k_start,
                 std::vector<double> hist_start,
                 size_t do_n_calculations)
{
    // convert lambda_0 and mu_0 to arama::mat
    auto X = array_to_mat(X_);
    auto lambda_0 = array_to_mat(lambda_0_);
    arma::rowvec mu_0 = arma::conv_to<arma::rowvec>::from(mu_0_);
    arma::uvec Z = arma::conv_to<arma::uvec>::from(Z_start);

    // X.print("X");
    // lambda_0.print("lambda_0");
    // mu_0.print("mu_0");
    // Z.print("Z");

    double Z_0 = calc_log_z(lambda_0, kappa_0, nu_0);

    std::vector<double> value_store(do_n_calculations);

    for(size_t i=0; i < do_n_calculations; ++i){
        // get data
        // update
        size_t num_cats = hist_start.size();
        double logp = 0;
        for(size_t k = 0; k < num_cats; ++k){
            arma::mat X_k = X.rows(arma::find(Z == k));
            if(X_k.is_empty()){
                for(auto val : hist_start)
                    std::cout << val << " ";
                std::cout << std::endl;
                Z.print("Z");
            }
            
            // int n_j = X_k.n_rows;
            logp += lgniwml(X_k, mu_0, lambda_0, kappa_0, nu_0, Z_0);
        }
        value_store[i] = logp + lcrp(hist_start, X.n_rows, crp_alpha);
        next_partition(Z, k_start, hist_start);
    }
    return logsumexp(value_store);
}

#endif