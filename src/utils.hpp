
#ifndef idsteach_cxx_utils_guard
#define idsteach_cxx_utils_guard

#include <armadillo>
#include <cmath>
#include <cfloat>
#include <vector>

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


static double logdet(arma::mat &A){
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