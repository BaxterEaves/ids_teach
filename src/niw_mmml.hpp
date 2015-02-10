// IDSTeach: Generate data to teach continuous categorical data.
// Copyright (C) 2015  Baxter S. Eaves Jr.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#ifndef idsteach_css_niwmmml_guard
#define idsteach_css_niwmmml_guard

#include <vector>
#include <armadillo>

#include "utils.hpp"
#include "debug.hpp"


static double calc_log_z(const arma::mat &lambda, double kappa, double nu)
{
    double d = double(lambda.n_rows);
    double log_z = 0;

    log_z = log(2)*(nu*d/2) + (d/2.0)*log(2*M_PI/kappa) + mvgammaln(nu/2, d) - (nu/2)*logdet(lambda);
    return log_z;
}


static void niw_update_params(const arma::mat &X, arma::rowvec &mu_0, arma::mat &lambda_0,
        double &kappa_0, double &nu_0)
{

    arma::mat xbar, S;

    auto n = static_cast<double>(X.n_rows); // number of data points
    auto d = X.n_cols; // dimensions

    // we have to take precautions because these functions perform
    // differently on row vectors, e.g., when there is only one
    // data point.
    if(n==1){
        xbar = X;
        S = arma::zeros(d, d);
    }else{
        S = arma::cov(X)*(n-1);
        xbar = arma::mean(X);
    }
    
    lambda_0 += S + (kappa_0*n)/(kappa_0+n)*trans((xbar-mu_0))*(xbar-mu_0);
    mu_0 = (kappa_0*mu_0 +n*xbar)/(kappa_0 + n);
    kappa_0 += n;
    nu_0 += n;
}


static double lgniwpp(const arma::mat &Y, const arma::mat &X, const arma::rowvec &mu_0,
        const arma::mat &lambda_0, double kappa_0, double nu_0)
{
    auto d = static_cast<double>(X.n_cols); // dimensions

    auto XY = join_vert(Y, X);

    auto lambda_n = lambda_0;
    auto mu_n = mu_0;
    auto kappa_n = kappa_0;
    auto nu_n = nu_0;

    auto lambda_m = lambda_0;
    auto mu_m = mu_0;
    auto kappa_m = kappa_0;
    auto nu_m = nu_0;

    niw_update_params(X, mu_n, lambda_n, kappa_n, nu_n);
    double Z_n = calc_log_z(lambda_n, kappa_n, nu_n);

    niw_update_params(XY, mu_m, lambda_m, kappa_m, nu_m);
    double Z_m = calc_log_z(lambda_m, kappa_m, nu_m);

    return Z_m - Z_n - log(2*M_PI)*(d/2);
}


static double lgniwpp(std::vector<std::vector<double>> Y, std::vector<std::vector<double>> X,
        std::vector<double> mu_0, std::vector<std::vector<double>> lambda_0, double kappa_0,
        double nu_0)
{
    auto Y_arma = array_to_mat(Y);
    auto X_arma = array_to_mat(X);
    auto lambda_0_arma = array_to_mat(lambda_0);
    arma::rowvec mu_0_arma = arma::conv_to<arma::rowvec>::from(mu_0);

    return lgniwpp(Y_arma, X_arma, mu_0_arma, lambda_0_arma, kappa_0, nu_0);
}


static double lgniwml(const arma::mat &X_k, const arma::rowvec &mu_0, const arma::mat &lambda_0, 
        double kappa_0, double nu_0, double Z_0)
{
    auto n = static_cast<double>(X_k.n_rows); // number of data points
    auto d = X_k.n_cols; // dimensions

    auto lambda_n = lambda_0;
    auto mu_n = mu_0;
    double kappa_n = kappa_0;
    double nu_n = nu_0;

    niw_update_params(X_k, mu_n, lambda_n, kappa_n, nu_n);

    double Z_n = calc_log_z(lambda_n, kappa_n, nu_n);

    return Z_n - Z_0 - log(2*M_PI)*(n*d/2);
} 


static double lgniwml(std::vector<std::vector<double>> &X_k, std::vector<double> &mu_0, 
        std::vector<std::vector<double>> &lambda_0, double kappa_0, double nu_0, double Z_0)
{
    auto X_k_arma = array_to_mat(X_k);
    auto lambda_0_arama = array_to_mat(lambda_0);
    arma::rowvec mu_0_arma = arma::conv_to<arma::rowvec>::from(mu_0);

    return lgniwml(X_k_arma, mu_0_arma, lambda_0_arama, kappa_0, nu_0, Z_0);
}


static double lgniwmmml(std::vector<std::vector<double>> X_,
        std::vector<std::vector<double>> lambda_0_, std::vector<double> mu_0_, double kappa_0,
        double nu_0, double crp_alpha, std::vector<size_t> Z_start, std::vector<size_t> k_start,
        std::vector<double> hist_start, size_t do_n_calculations)
{
    // convert lambda_0 and mu_0 to arama::mat
    auto X = array_to_mat(X_);
    auto lambda_0 = array_to_mat(lambda_0_);
    arma::rowvec mu_0 = arma::conv_to<arma::rowvec>::from(mu_0_);
    arma::uvec Z = arma::conv_to<arma::uvec>::from(Z_start);

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


static double score_data(const std::vector<size_t> &Z, const arma::mat &X,
        const arma::rowvec &mu_0, const arma::mat &lambda_0, double kappa_0, double nu_0,
        double log_z)
{
    size_t K = *max_element(Z.begin(), Z.end());
    double score = 0;
    for(size_t k = 0; k <= K; ++k){
        std::vector<size_t> idx;
        for(size_t i = 0; i < Z.size(); ++i)
            if(Z[i] == k) idx.push_back(i);
        auto X_k = fetch_rows(X, idx);
        score += lgniwml(X_k, mu_0, lambda_0, kappa_0, nu_0, log_z);
    }
    return score;
}

#endif