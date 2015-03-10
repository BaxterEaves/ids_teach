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


#include "dpgmm.hpp"

using std::vector;


void DPGMM::fit(size_t n, double sm_prop, size_t num_sm_sweeps, size_t sm_burn)
{
    std::uniform_real_distribution<double> rand(0, 1);

    for(size_t i = 0; i < n; ++i){
        if(i < sm_burn or rand(_rng) < sm_prop){
            __update_sm(num_sm_sweeps);
        } else {
            __update_gibbs();
        }
    }
}


vector<size_t> DPGMM::predict(arma::mat Y)
{ 
    vector<size_t> prediction;

    for(size_t i = 0; i < Y.n_rows; ++i){
        arma::mat y = Y.row(i);
        ASSERT_EQUAL(std::cout, y.n_rows, 1);
        ASSERT_EQUAL(std::cout, y.n_cols, _X.n_cols);

        std::vector<double> ps(_Nk.begin(), _Nk.end());
        for(auto & p : ps) p = log(p);

        for(size_t k = 0; k < _K; ++k){
            std::vector<size_t> idx;
            for(size_t i = 0; i < _n; ++i) if (_Z[i] == k) idx.push_back(i);

            auto Xk = fetch_rows(_X, idx);
            ASSERT_EQUAL(std::cout, Xk.n_rows, _Nk[k]);
            ps[k] += lgniwpp(y, Xk, _mu_0, _lambda_0, _kappa_0, _nu_0);
        }

        size_t k = argmax(ps);
        ASSERT(std::cout, k < _K);
        prediction.push_back(k);
    }
    return prediction;
}


vector<size_t> DPGMM::predict(vector<vector<double>> Y)
{
    return predict(array_to_mat(Y));
}


// ________________________________________________________________________________________________
// Split-merge items
// ````````````````````````````````````````````````````````````````````````````````````````````````
void DPGMM::__update_sm(size_t num_sm_sweeps){
    ASSERT_GREATER_THAN_ZERO(std::cout, num_sm_sweeps);

    std::uniform_int_distribution<size_t> row_rand(0, _n-1);
    std::uniform_real_distribution<double> rand(0.0, 1.0);

    size_t row_1, row_2;
    row_1 = row_rand(_rng);
    ASSERT(std::cout, row_1 >= 0 and row_1 < _n);

    unsigned int try_iters = 0;
    do{
        row_2 = row_rand(_rng);
        ASSERT(std::cout, row_2 >= 0 and row_2 < _n);
        ++try_iters;
        if(try_iters > _n){
            std::cout << "Could not draw two unique rows among " << _n << "..." << std::endl;
            std::cout << "First row was " << row_1 << "." << std::endl;
            throw 1;
        }
    }while(row_1 == row_2);

    auto k1 = _Z[row_1];
    auto k2 = _Z[row_2];

    vector<size_t> sweep_indices;
    for(size_t i = 0; i < _n; ++i)
        if((_Z[i] == k1 or _Z[i] == k2) and (i != row_1 and i != row_2)) 
            sweep_indices.push_back(i);

    auto Z_launch = _Z;
    auto Nk_launch = _Nk;

    auto k1_launch = k1;
    if(k1 == k2){
        k1_launch = _K;
        Z_launch[row_1] = k1_launch;
        --Nk_launch[k1];
        Nk_launch.push_back(1);
    }

    auto k2_launch = k2;
    
    __restricted__init(Z_launch, Nk_launch, k1_launch, k2_launch, sweep_indices);

    double Q_launch = 0;
    for(size_t sweep = 0; sweep < num_sm_sweeps; ++sweep){
        Q_launch = __restricted_gibbs_sweep(Z_launch, Nk_launch, k1_launch, k2_launch,
                                            sweep_indices);
    }

    ASSERT(std::cout, Q_launch != 0);

    if(k1 == k2){
        // split propsal
        double score_split = score_data(Z_launch, _X, _mu_0, _lambda_0, _kappa_0, _nu_0, _log_z);
        double score_current = score_data(_Z, _X, _mu_0, _lambda_0, _kappa_0, _nu_0, _log_z);
        double L = score_split - score_current;
        double P = lcrp(Nk_launch, _n, _crp_alpha) - lcrp(_Nk, _n, _crp_alpha);

        if(log(rand(_rng)) < P+L-Q_launch){
            _Z = Z_launch;
            _Nk = Nk_launch;
            _K = Nk_launch.size();
        }

    }else{
        // merge proposal
        auto Z_merge = _Z;
        auto Nk_merge =_Nk;
        auto k1_merge = k2;
        auto k2_merge = k2;

        Z_merge[row_1] = k2;
        for(auto &idx : sweep_indices)
            Z_merge[idx] = k2;

        Nk_merge[k2] += Nk_merge[k1];
        Nk_merge.erase(Nk_merge.begin() + k1);
        for(auto &z : Z_merge)
            if(z > k1) --z;

        double Q = __restricted_gibbs_sweep(Z_launch, Nk_launch, k1_launch, k2_launch,
                                            sweep_indices, _Z);

        ASSERT(std::cout, Q != 0);
        
        double score_merge = score_data(Z_merge, _X, _mu_0, _lambda_0, _kappa_0, _nu_0, _log_z);
        double score_current = score_data(_Z, _X, _mu_0, _lambda_0, _kappa_0, _nu_0, _log_z);
        double L = score_merge - score_current;
        double P = lcrp(Nk_merge, _n, _crp_alpha) - lcrp(_Nk, _n, _crp_alpha);

        if(log(rand(_rng)) < P+Q+L){
            _Z = Z_merge;
            _Nk = Nk_merge;
            _K = Nk_merge.size();
        }
    }
    
}


void DPGMM::__restricted__init(vector<size_t> &Z, vector<double> &Nk, size_t k1, size_t k2,
    vector<size_t> sweep_indices)
{
    std::uniform_real_distribution<double> rand(0.0, 1.0);

    vector<size_t> idx_1 = {k1};
    vector<size_t> idx_2 = {k2};

    Nk[k1] = 1;
    Nk[k2] = 1;

    for(auto row : sweep_indices){
        auto Y = _X.row(row);
        auto X_1 = fetch_rows(_X, idx_1);
        auto X_2 = fetch_rows(_X, idx_2);

        double lp_k1 = log(Nk[k1]) + lgniwpp(Y, X_1, _mu_0, _lambda_0, _kappa_0, _nu_0);
        double lp_k2 = log(Nk[k2]) + lgniwpp(Y, X_2, _mu_0, _lambda_0, _kappa_0, _nu_0);

        auto C = logsumexp({lp_k1, lp_k2});

        if(log(rand(_rng)) < lp_k1-C){
            ++Nk[k1];
            Z[row] = k1;
            idx_1.push_back(row);
        }else{
            ++Nk[k2];
            Z[row] = k2;
            idx_2.push_back(row);
        }
    }
    ASSERT_EQUAL(std::cout, *max_element(Z.begin(), Z.end()), Nk.size()-1);
    ASSERT_EQUAL(std::cout, std::accumulate(Nk.begin(), Nk.end(), 0), Z.size());
}


double DPGMM::__restricted_gibbs_sweep(vector<size_t> &Z, vector<double> &Nk, size_t k1, size_t k2,
    vector<size_t> sweep_indices, const vector<size_t> &Z_final)
{
    std::uniform_real_distribution<double> rand(0.0, 1.0);

    double lp_split = 0;
    bool is_hypothetical = not Z_final.empty();

    for(auto row : sweep_indices){
        auto k_a = Z[row];

        double lcrp_1, lcrp_2;
        if(k_a == k1){
            lcrp_1 = log(Nk[k1]-1.0);
            lcrp_2 = log(Nk[k2]);
        }else{
            lcrp_1 = log(Nk[k1]);
            lcrp_2 = log(Nk[k2]-1.0);
        }

        vector<size_t> idx_1, idx_2;
        for(size_t i = 0; i < _n; ++i){
            if(Z[i] == k1 and i != row) idx_1.push_back(i);
            if(Z[i] == k2 and i != row) idx_2.push_back(i);
        }

        auto Y = _X.row(row);
        auto X_1 = fetch_rows(_X, idx_1);
        auto X_2 = fetch_rows(_X, idx_2);

        double lp_k1 = lcrp_1 + lgniwpp(Y, X_1, _mu_0, _lambda_0, _kappa_0, _nu_0);
        double lp_k2 = lcrp_2 + lgniwpp(Y, X_2, _mu_0, _lambda_0, _kappa_0, _nu_0);

        auto C = logsumexp({lp_k1, lp_k2});

        size_t k_b;
        if(is_hypothetical){
            k_b = Z_final[row];
        }else{
            k_b = (log(rand(_rng)) < lp_k1 - C) ? k1 : k2;
        }

        if(k_a != k_b){
            Z[row] = k_b;
            --Nk[k_a];
            ++Nk[k_b];
        }

        if(k_b == k1){
            lp_split += lp_k1 - C;
        } else{
            lp_split += lp_k2 - C;
        }
    }
    return lp_split;
}


// ________________________________________________________________________________________________
// Gibbs items
// ````````````````````````````````````````````````````````````````````````````````````````````````
void DPGMM::__update_gibbs()
{
    std::shuffle(_row_list.begin(), _row_list.end(), _rng);
    for(auto row : _row_list){
        arma::mat Y = _X.row(row);
        ASSERT_EQUAL(std::cout, Y.n_rows, 1);
        ASSERT_EQUAL(std::cout, Y.n_cols, _X.n_cols);

        size_t k_a = _Z[row];
        bool is_singleton = (_Nk[k_a] == 1);

        ASSERT_GREATER_THAN_ZERO(std::cout, _Nk[k_a]);

        std::vector<double> ps(_Nk.begin(), _Nk.end());
        if(not is_singleton){
            ps.push_back(_crp_alpha);
        }else{
            ps[k_a] = _crp_alpha;
        }

        for(auto & p : ps) p = log(p);

        if(not is_singleton){
            std::vector<size_t> idx;
            for(size_t i = 0; i < _Z.size(); ++i) if (_Z[i] == k_a and i != row) idx.push_back(i);
            // remove from current component and calculate p
            auto Xk = fetch_rows(_X, idx);
            ASSERT_EQUAL(std::cout, Xk.n_rows, _Nk[k_a]-1);
            ps[k_a] += lgniwpp(Y, Xk, _mu_0, _lambda_0, _kappa_0, _nu_0);
            // singleton prob
            ps.back() += lgniwml(Y, _mu_0, _lambda_0, _kappa_0, _nu_0, _log_z);
        }else{
            ps[k_a] += lgniwml(Y, _mu_0, _lambda_0, _kappa_0, _nu_0, _log_z);
        }

        for(size_t k_b = 0; k_b < _K; ++k_b){
            if(k_b != k_a){
                std::vector<size_t> idx;
                for(size_t i = 0; i < _n; ++i) if (_Z[i] == k_b) idx.push_back(i);

                auto Xk = fetch_rows(_X, idx);
                ASSERT_EQUAL(std::cout, Xk.n_rows, _Nk[k_b]);
                ps[k_b] += lgniwpp(Y, Xk, _mu_0, _lambda_0, _kappa_0, _nu_0);
            }
        }

        size_t k_b = lpflip(ps, _rng);
        ASSERT(std::cout, k_b <= _K);

        // clean up
        if(k_b != k_a){
            _Z[row] = k_b;
            if(is_singleton){
                ++_Nk[k_b];
                _Nk.erase(_Nk.begin() + k_a);
                for(auto &z : _Z) if(z >= k_a) --z;
            }else{
                --_Nk[k_a];
                if(k_b == _K){
                    _Nk.push_back(1);
                }else{ 
                    ++_Nk[k_b];
                }
            }
        }
 
        _K = _Nk.size();
        ASSERT_EQUAL(std::cout, *max_element(_Z.begin(), _Z.end()), _Nk.size()-1);
        ASSERT_EQUAL(std::cout, std::accumulate(_Nk.begin(), _Nk.end(), 0), _Z.size());
    }
}
