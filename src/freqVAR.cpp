#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List freqVAR(arma::mat series, int p, bool include_mean = true) {
  
  int k = series.n_cols, n = series.n_rows;
  
  arma::mat Y = series.rows(p, n - 1);
  arma::mat X = series.rows(p - 1, n - 2);
  arma::mat X_1;
    
  for (int i = 2; i < p + 1; ++i) {
    X_1 = series.rows(p - i, n - 1 - i);
    X = join_rows(X, X_1);
  }
  
  if (include_mean) {
    arma::vec intcpt(n - p) ;
    intcpt.fill(1) ;
    X = join_rows(intcpt, X) ;
  }
  
  arma::mat coef = ((inv(X.t() * X) * X.t()) * Y).t();
  arma::mat res = Y - X*coef.t();
  arma::mat covmat = cov(res);
  
  return List::create(Named("Y") = Y,
                      Named("X") = X,
                      Named("Coefficients") = coef,
                      Named("Residuals") = res,
                      Named("CovMat") = covmat,
                      Named("Lags") = p, 
                      Named("Variables") = k);
}