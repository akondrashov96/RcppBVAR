#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List BVAR(arma::mat series, int p, 
          arma::mat C, arma::mat V0, arma::mat Phi0,
          int df = 5, bool include_mean = true) {
  
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
  
  Rcout << X << endl;
  
  arma::mat Phi = inv(X.t() * X + C) * (X.t() * Y + C * Phi0);
  arma::mat res = Y - X*Phi;
  arma::mat covmat = cov(res);
  
  arma::mat SD = kron((V0 + res.t() * res + (Phi - Phi0).t() * C * (Phi - Phi0)) / (df + n - p - k - 1), 
                      (inv(X.t() * X + C))) ;
  arma::vec se = sqrt(SD.diag()) ;
  arma::mat Coefs = join_rows(vectorise(Phi), se);
  Coefs = join_rows(Coefs, vectorise(Phi)/se);
  
  return List::create(Named("Y") = Y,
                      Named("X") = X,
                      Named("Phi") = Phi,
                      Named("Coefficients") = Coefs,
                      Named("Residuals") = res,
                      Named("CovMat") = covmat,
                      Named("Lags") = p, 
                      Named("Variables") = k,
                      Named("Phi_prior") = Phi0,
                      Named("Precision_mat") = C,
                      Named("V0_prior") = V0);
  
}