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


#ifndef idsteach_cxx_utils_guard
#define idsteach_cxx_utils_guard

#include <armadillo>
#include <cmath>
#include <cfloat>
#include <cmath>
#include <vector>
#include <random>

#include "debug.hpp"


// log(Pi)
#define LOG_PI 1.1447298858494001638774761886452324688434600830078125

// log(Pi)/4
#define LOG_PI4 0.286182471462350040969369047161308117210865020751953125

// log(2)
#define LOG_2 0.69314718055994528622676398299518041312694549560546875

// Log(2*pi)
#define LOG_2PI 1.83787706640934533908193770912475883960723876953125


static arma::mat array_to_mat(std::vector<std::vector<double>> X){
    arma::mat out = arma::zeros(X.size(), X[0].size());
    for(size_t i = 0; i < X.size(); ++i)
        for(size_t j = 0; j < X[0].size(); ++j)
            out(i, j) = X[i][j];
    return out;
}


static arma::mat fetch_rows(const arma::mat &X, const std::vector<size_t> &idx){
    ASSERT(std::cout, idx.size() > 0);
    arma::mat out(idx.size(), X.n_cols);
    for(size_t i = 0; i  < idx.size(); ++i)
        out.row(i) = X.row(idx[i]);

    ASSERT(std::cout, out.n_rows == idx.size());
    ASSERT_EQUAL(std::cout, out.n_cols, X.n_cols);
    return out;
}


static double mvgammaln(double a, unsigned int d){
    double l;

    arma::vec A = arma::linspace(1, d, d);
    A = a+((1-A)*(.5));

    double gammaSum = 0;

    for(unsigned int i = 0; i < d; i++){
        gammaSum += lgamma(A(i));
    }

    l = d*(d-1)*LOG_PI4 + gammaSum;

    return l;
}


static double logdet(const arma::mat &A){
    double val;
    if(A.n_rows == 2){
        double a,b,c;

        a = sqrt(A.at(0,0));
        b = A.at(1,0)/a;
        c = A.at(1,1);

        val = 2*(log(a) + .5*log(c-b*b));
    }else{
        double sign;
        log_det(val, sign, A);
    }
    return val;
}


static double logsumexp(std::vector<double> P)
{
    // if there is a single element in the vector, return that element
    // otherwise there will be log domain problems
    if(P.size() == 1)
        return P.front();

    double max = *std::max_element(P.begin(), P.end());
    double ret = 0;
    for(size_t i = 0; i < P.size(); ++i)
        ret += exp(P[i]-max);

    double retval = log(ret)+max;
    ASSERT_IS_A_NUMBER(std::cout, retval);
    return retval;
}


// log crp probability
static double lcrp(const std::vector<double> &Nk, size_t n, double alpha)
{
    ASSERT_GREATER_THAN_ZERO(std::cout, alpha);
    ASSERT_GREATER_THAN_ZERO(std::cout, n);

    double K = double(Nk.size());
    double sum_gammaln = 0;

    for(auto k : Nk)
        sum_gammaln += lgamma(k);

    return sum_gammaln + K*log(alpha) + lgamma(alpha) - lgamma(double(n)+alpha);
}


// multinomial draw from a vector of log probabilities
static size_t lpflip(std::vector<double> P, std::mt19937 &rng)
{
    // normalize
    double norm_const = logsumexp(P);
    std::uniform_real_distribution<double> dist(0, 1);
    double r = dist(rng);
    double cumsum = 0;
    for(size_t i = 0; i < P.size(); i++){
        cumsum += exp(P[i]-norm_const);
        if( r < cumsum)
            return i;
    }

    throw 2;
}


// constructs a parition, Z, with K categories, and counts, Nk, from CRP(alpha)
static void crpGen(double alpha, size_t N, std::vector<size_t> &Z, size_t &K,
        std::vector<double> &Nk, std::mt19937 &rng)
{
    // setup
    Z.resize(N);
    Nk = {1};
    K = 1;

    double log_alpha = log(alpha);
    double denom = alpha + 1;

    std::vector<double> logps(2, 0);

    for(size_t i = 1; i < N; ++i){
        double log_denom = log(denom);
        for( size_t k = 0; k < K; ++k)
            logps[k] = log(Nk[k]) - log_denom;
        logps.back() = log_alpha - log_denom;

        size_t z = lpflip(logps, rng);

        Z[i] = z;
        // if a new category has been added, add elements where needed
        if(z == K){
            logps.push_back(0);
            Nk.push_back(1);
            ++K;
        }else{
            ++Nk[z];
        }

        ++denom;

        ASSERT(std::cout, Nk.size() == K);
    }
}

// Builds the next partition Z, and supplemental counts k. expects Z and k to be started at 
// {0,..,0}.
static bool next_partition(arma::uvec &Z, std::vector<size_t> &k, std::vector<double> &hist)
{
    auto n = Z.n_elem;
    for(size_t i = n-1; i >= 1; --i){
        if(Z(i) <= k[i-1]){
            --hist[Z(i)];
            ++Z(i);

            if(Z(i) == hist.size()) hist.push_back(1); else ++hist[Z(i)];
            if(k[i] <= Z(i)) k[i] = Z(i);

            for (size_t j = i+1; j <= n-1; ++j){
                --hist[Z(j)];
                ++hist[Z(0)];

                Z(j) = Z(0);
                k[j] = k[i];
            }
            while(hist.back() == 0)
                hist.pop_back();
            return true;
        }
    }
    return false;
}

#endif