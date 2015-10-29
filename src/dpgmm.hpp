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
          double nu_0,  double crp_alpha, bool single_cluster_init, int seed) : 
    _X(X), _mu_0(mu_0), _lambda_0(lambda_0), _kappa_0(kappa_0), _nu_0(nu_0),
    _crp_alpha(crp_alpha), _n(X.n_rows)
    {
        if ( seed >= 0 ){
            _rng = std::mt19937(seed); 
        } else {
            std::random_device rd;
            _rng = std::mt19937(rd());
        }

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

        _logp_itr.push_back(logp());
        _K_itr.push_back(_K);

        _Z_max = _Z;
        _lp_max = logp();
    }

    DPGMM(std::vector<std::vector<double>> X, std::vector<double> mu_0, 
          std::vector<std::vector<double>> lambda_0, double kappa_0,
          double nu_0, double crp_alpha, bool single_cluster_init, int seed) :
     _kappa_0(kappa_0), _nu_0(nu_0), _crp_alpha(crp_alpha), _n(X.size()),
     _X(array_to_mat(X)), _lambda_0(array_to_mat(lambda_0)),
     _mu_0(arma::conv_to<arma::rowvec>::from(mu_0))
     {
        if ( seed >= 0 ){
            _rng = std::mt19937(seed); 
        } else {
            std::random_device rd;
            _rng = std::mt19937(rd());
        }

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

        _logp_itr.push_back(logp());
        _K_itr.push_back(_K);

        _Z_max = _Z;
        _lp_max = logp();
    }
    double seqinit(size_t n_sweeps); 

    void set_assignment(std::vector<size_t> asgmnt);

    double fit(size_t n_iter, double sm_prop, size_t num_sm_sweeps,
             size_t sm_burn);

    std::vector<size_t> predict(std::vector<std::vector<double>> Y);
    std::vector<size_t> predict(arma::mat Y);

    std::vector<size_t> get_Z(){return _Z;};
    std::vector<double> get_Nk(){return _Nk;};
    size_t get_K(){return _K;};

    std::vector<double> get_logps(){return _logp_itr;};
    std::vector<size_t> get_ks(){return _K_itr;};
    std::vector<size_t> get_z_max(){return _Z_max;};
    double get_logp_max(){return _lp_max;};

    double logp();

private:
    // Does iterative Gibbs on each row in random order
    void __update_gibbs(double &logp_trns);
    void __update_sm(size_t num_sm_sweeps, double &logp_trns);
    void __restricted__init(std::vector<size_t> &Z, std::vector<double> &Nk,
                            size_t k1, size_t k2,
                            std::vector<size_t> sweep_indices);
    double __restricted_gibbs_sweep(std::vector<size_t> &Z,
                                    std::vector<double> &Nk, size_t k1,
                                    size_t k2,
                                    std::vector<size_t> sweep_indices,
                                    const std::vector<size_t> &Z_final={});

    std::mt19937 _rng;              

    // the data
    const arma::mat _X;
    // number of data points
    const size_t _n;
    // row indices for changing update order
    std::vector<size_t> _row_list;  

    // prior mean 
    const arma::rowvec _mu_0;
    // prior scal matrix
    const arma::mat _lambda_0;
    // number of prior observations        
    const double _kappa_0;
    // degrees of freedom
    const double _nu_0;
    // CRP discount parameter
    const double _crp_alpha;
    // default normalizing constant
    double _log_z;

    // _Z[i] is the component to which datum i belongs
    std::vector<size_t> _Z;
    // _Nk[k] is the number of data assigned to component k
    std::vector<double> _Nk;
    // number of components
    size_t _K;

    // probability of the data given the partitioning, P(X|Z) for each
    // iterations
    std::vector<double> _logp_itr;
    // Number of components for each iteration
    std::vector<size_t> _K_itr;

    // highest logp
    double _lp_max;
    // best assignment vector
    std::vector<size_t> _Z_max;

};

#endif
