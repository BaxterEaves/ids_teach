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

#ifndef idsteach_css_dpgmm_guard
#define idsteach_css_dpgmm_guard

#include <vector>
#include <algorithm>
#include <armadillo>
#include <random>

#include "utils.hpp"
#include "debug.hpp"
#include "niw_mmml.hpp"

class DPGMM
{
public:
    DPGMM(arma::mat X, arma::rowvec mu_0, arma::mat lambda_0, double kappa_0,
            double nu_0,  double crp_alpha, bool single_cluster_init) : 
    _X(X), _mu_0(mu_0), _lambda_0(lambda_0), _kappa_0(kappa_0), _nu_0(nu_0), _crp_alpha(crp_alpha),
    _n(X.n_rows)
    {
        if(single_cluster_init){
            _Z.resize(_n, 0);
            _Nk.resize(1, _n);
            _K = 1;
        }else{
            crpGen(_crp_alpha, _n, _Z, _K, _Nk, _rng);
        }
        _log_z = calc_log_z(_lambda_0, _kappa_0, _nu_0);

        _row_list.resize(_n);
        for(size_t i = 0; i < _n; ++i) _row_list[i] = i;
    }

    // to be called from the python wrapper
    DPGMM(std::vector<std::vector<double>> X, std::vector<double> mu_0, 
            std::vector<std::vector<double>> lambda_0, double kappa_0, double nu_0,
            double crp_alpha, bool single_cluster_init) :
     _kappa_0(kappa_0), _nu_0(nu_0), _crp_alpha(crp_alpha), _n(X.size()), _X(array_to_mat(X)),
     _lambda_0(array_to_mat(lambda_0)), _mu_0(arma::conv_to<arma::rowvec>::from(mu_0))
     {
        if(single_cluster_init){
            _Z.resize(_n, 0);
            _Nk.resize(1, _n);
            _K = 1;
        }else{
            crpGen(_crp_alpha, _n, _Z, _K, _Nk, _rng);
        }
        _log_z = calc_log_z(_lambda_0, _kappa_0, _nu_0);

        _row_list.resize(_n);
        for(size_t i = 0; i < _n; ++i) _row_list[i] = i;
    }
    
    // Does n Gibbs speeps
    void fit(size_t n_iter, double sm_prop, size_t num_sm_sweeps, size_t sm_burn);

    // Returns the assignment vector
    std::vector<size_t> get_Z(){return _Z;};

private:
    // Does iterative Gibbs on each row in random order
    void __update_gibbs();
    void __update_sm(size_t num_sm_sweeps);
    void __restricted__init(std::vector<size_t> &Z, std::vector<double> &Nk, size_t k1, size_t k2,
        std::vector<size_t> sweep_indices);
    double __restricted_gibbs_sweep(std::vector<size_t> &Z, std::vector<double> &Nk, size_t k1,
        size_t k2, std::vector<size_t> sweep_indices, const std::vector<size_t> &Z_final={});

    // random number generator
    std::mt19937 _rng;              

    // the data
    const arma::mat _X;
    const size_t _n;                // number of data points
    std::vector<size_t> _row_list;  // row indices for changing update order

    // NIW prior (see Murphy [2007])
    const arma::rowvec _mu_0;       // prior mean
    const arma::mat _lambda_0;      // prior scal matrix
    const double _kappa_0;          // number of prior observations
    const double _nu_0;             // degrees of freedom
    const double _crp_alpha;        // CRP discount parameter
    double _log_z;                  // default normalizing constant

    // partition information
    std::vector<size_t> _Z;         // _Z[i] is the component to which datum i belongs
    std::vector<double> _Nk;        // _Nk[k] is the number of data assigned to component k
    size_t _K;                      // number of components

};

#endif