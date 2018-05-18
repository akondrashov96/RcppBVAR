#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List Forecast(List model, int h) {
  
  int p = model["Lags"], k = model["Variables"];
  arma::mat y_hat = mat(k, h);
  
  NumericMatrix Y = model["Y"];
  int n = Y.nrow();
  arma::mat Y_init = Y(Range(n - p, n - 1), Range(0, k - 1));
  Y_init = reverse(Y_init, 0);
  
  arma::mat coefmat = model["Coefficients"];
  
  for (int i = 0; i < h; ++i) {
    Rcout << i << endl << Y_init << endl;
    y_hat.col(i) = coefmat * vectorise(Y_init.t());
    Y_init = join_cols(y_hat.col(i).t(), 
                       Y_init.rows(0, p - 2));
  }
  
  return List::create(Named("y_hat") = y_hat.t(),
                      Named("Y_init") = Y_init);
}