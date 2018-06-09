#ifndef prepData_h
#define prepData_h

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

List prepData(arma::mat series, int p, bool include_const = true);
  
#endif