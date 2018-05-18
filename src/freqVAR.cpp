#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List freqVAR(NumericMatrix series, int p) {
  
  int k = series.ncol(), n = series.nrow();
  
  arma::mat Y = series(Range(p, n - 1), Range(0, k - 1));
  arma::mat X = series(Range(p - 1, n - 2), Range(0, k - 1));
  arma::mat X_1;
    
  for (int i = 2; i < p + 1; ++i) {
    X_1 = series(Range(p - i, n - 1 - i), Range(0, k - 1));
    X = join_rows(X, X_1);
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