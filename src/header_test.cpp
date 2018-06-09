#include <RcppArmadillo.h>
//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]
#include "prepdata.h"

using namespace Rcpp;
using namespace arma;

List prepData(arma::mat series, int p, bool include_const = true) {
  
  int k = series.n_cols, n = series.n_rows;
  
  arma::mat Y = series.rows(p, n - 1);
  arma::mat X = series.rows(p - 1, n - 2);
  arma::mat X_1;
  
  for (int i = 2; i < p + 1; ++i) {
    X_1 = series.rows(p - i, n - 1 - i);
    X = join_rows(X, X_1);
  }
  
  if (include_const) {
    arma::vec intcpt(n - p) ;
    intcpt.fill(1) ;
    X = join_rows(X, intcpt) ;
  }
  
  return List::create(Named("Y") = Y,
                      Named("X") = X,
                      Named("k") = k,
                      Named("n") = n,
                      Named("p") = p);
}