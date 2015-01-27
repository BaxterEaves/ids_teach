#include <armadillo>
#include <cmath>

#include "niw_mmml.hpp"
#include "utils.hpp"

int main(){
    arma::mat X_single, X_multi;
    arma::mat L0_generic, L0_random;
    arma::rowvec M0_generic, M0_random;
    double K0_generic, K0_random;
    double V0_generic, V0_random;

    X_single << 3.57839693972576 << 0.725404224946106 << arma::endr;

    X_multi << 3.57839693972576 << 0.725404224946106 << arma::endr
            << 2.76943702988488 << -0.0630548731896562 << arma::endr
            << -1.34988694015652 << 0.714742903826096 << arma::endr
            << 3.03492346633185 << -0.204966058299775 << arma::endr;

    L0_generic << 1.0 << 0.0 << arma::endr << 0.0 << 1.0 << arma::endr;
    L0_random << 0.226836817541677 << -0.0200753958619398 << arma::endr
              << -0.0200753958619398 << 0.217753683861863 << arma::endr;

    M0_generic << 0.0 << 0.0 << arma::endr;
    M0_random << -0.124144348216312 << 1.48969760778546 << arma::endr;

    K0_generic = 1.0;
    V0_generic = 2.0;

    K0_random = 2.03620546457332;
    V0_random = 2.273220391735;

    //=============================================================================================
    double mvgam;
    mvgam = mvgammaln(1, 1);
    std::cout << "mvgammaln(1, 1), should be " << 0.0 
        << ", is " << mvgam << std::endl;
    mvgam = mvgammaln(11, 1);
    std::cout << "mvgammaln(11, 1), should be " << 15.1044125730755 
        << ", is " << mvgam << std::endl;
    mvgam = mvgammaln(2.32, 2);
    std::cout << "mvgammaln(2.32, 2), should be " << 0.673425977668427
        << ", is " << mvgam << std::endl;

    //=============================================================================================
    double ldet;
    ldet =  logdet(L0_generic);
    std::cout << "logdet(eye(2)), should be " << 0.0 
        << ", is " << ldet << std::endl;
    ldet =  logdet(L0_random);
    std::cout << "logdet(random), should be " << -3.01610782976437 
        << ", is " << ldet << std::endl;
    
    //=============================================================================================
    double log_z;
    log_z = calc_log_z(L0_generic, K0_generic, V0_generic);
    std::cout << "calc_log_z(generic), should be " << 4.3689013133786361 
        << ", is " << log_z << std::endl;
    log_z = calc_log_z(L0_random, K0_random, V0_random);
    std::cout << "calc_log_z(random), should be " << 6.9827188731298246 
        << ", is " << log_z << std::endl;

    //=============================================================================================
    double logp, Z_0;
    Z_0 = calc_log_z(L0_generic, K0_generic, V0_generic);
    logp = lgniwml(X_single, M0_generic, L0_generic, K0_generic, V0_generic, Z_0);
    std::cout << "generic marginal (single), should be " << -5.5861321608291
              << ", is " << logp << std::endl;

    logp = lgniwml(X_multi, M0_generic, L0_generic, K0_generic, V0_generic, Z_0);
    std::cout << "generic marginal (mulit), should be " << -16.3923777220275
              << ", is " << logp << std::endl;


    Z_0 = calc_log_z(L0_random, K0_random, V0_random);
    logp = lgniwml(X_single, M0_random, L0_random, K0_random, V0_random, Z_0);
    std::cout << "random marginal (single), should be " << -6.60964751885643
              << ", is " << logp << std::endl;

    logp = lgniwml(X_multi, M0_random, L0_random, K0_random, V0_random, Z_0);
    std::cout << "random marginal (mulit), should be " << -19.5739755706395
              << ", is " << logp << std::endl;

}