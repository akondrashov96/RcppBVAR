#include <RcppArmadillo.h>

//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

//[[Rcpp::export]]
List genVAR(int nT, int p, int k) {
  
  if (p > nT) {
    Rcpp::stop("Number of lags cannot exceed the number of observations");
  }
  
  List mats;
  

  arma::mat Phi(k, p * k);
  for (int row = 0 ; row < Phi.n_rows ; ++row) {
    Phi.row(row) = as<vec>(Rcpp::rnorm(p*k, 0, 10/nT)).t();
  }
  arma::mat series = zeros<mat>(k, nT + 2*p);
  
  for (int i = p ; i < nT + p ; ++i) {
    for (int j = 0 ; j < p ; ++j) {
      series.col(i) = series.col(i) + Phi.cols(k*j, k*(j + 1) - 1) * series.col(i - 1 - j);
    }
    series.col(i) = series.col(i) + randn<vec>(k, 1);
  }
  
  series = series.cols(p, nT + p - 1).t();
  return List::create(Named("series") = series,
                      Named("Phi") = Phi,
                      Named("p") = p);
}