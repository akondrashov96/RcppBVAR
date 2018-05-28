#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
double AR1(arma::vec series, int p = 1) {
  
  int n = series.n_elem;
  
  arma::vec Y = series.rows(p, n - 1);
  arma::mat X = series.rows(p - 1, n - 2);
  arma::vec intcpt(n - p) ;
  intcpt.fill(1) ;
  X = join_rows(X, intcpt) ;
  
  arma::mat coef = ((inv(X.t() * X) * X.t()) * Y).t();
  double res = as_scalar(coef(0,0));
  
  return res;
}

//[[Rcpp::export()]]
arma::vec conj_delta(arma::mat series, arma::vec delt, bool AR1type = false) {
  
  int m = series.n_cols, n = series.n_rows;
  arma::vec delta;
  delta = delt;
  
  try {
    if (AR1type) {
      delta.set_size(m);
      delta.fill(1);
      arma::vec y_uni(n);
      
      for (int i = 0 ; i < m ; ++i) {
        Rcout << i << endl;
        
        y_uni = series.col(i);
        delta[i] = AR1(y_uni);
        if (delta[i] > 1) {
          delta[i] = 1;
        }
      }
    } else if (delt.n_elem == 1) {
      delta.set_size(m);
      delta.fill(delt(0));
    } else if (delt.n_elem != m) {
      throw std::range_error("Length of delta should be equal to 1 or m");
    }
  } catch(std::exception &ex) {	
    forward_exception_to_r(ex);
  } catch(...) { 
    ::Rf_error("C++ exception (unknown reason)"); 
  }
  
  return delta;
}