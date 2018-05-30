#include <RcppArmadillo.h>

//[[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// creates X and Y matrices from provided series
//[[Rcpp::export()]]
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
    X = join_rows(intcpt, X) ;
  }
  
  return List::create(Named("Y") = Y,
                      Named("X") = X,
                      Named("k") = k,
                      Named("n") = n,
                      Named("p") = p);
}

//estimate AR1 model for delta func
// [[Rcpp::export]]
List ARp(arma::vec series, int p = 1) {
  
  int n = series.n_elem;
  
  arma::vec Y = series.rows(p, n - 1);
  arma::mat X = series.rows(p - 1, n - 2);
  if (p > 1) {
    arma::mat X_1;
    for (int i = 2; i < p + 1; ++i) {
      X_1 = series.rows(p - i, n - 1 - i);
      X = join_rows(X, X_1);
    }
  }
  arma::vec intcpt(n - p) ;
  intcpt.fill(1) ;
  X = join_rows(X, intcpt) ;
  
  
  arma::mat coefs = ((inv(X.t() * X) * X.t()) * Y).t();
  double AR_1 = as_scalar(coefs(0,0));
  
  arma::vec resid = Y - X * coefs.t();
  
  return List::create(Named("AR1") = AR_1,
                      Named("res") = resid);
}

//[[Rcpp::export()]]
arma::vec conj_delta(arma::mat series, arma::vec delt, bool deltAR1type = false) {
  
  int m = series.n_cols, n = series.n_rows;
  arma::vec delta;
  delta = delt;
  List coef;
  
  try {
    if (deltAR1type) {
      delta.set_size(m);
      delta.fill(1);
      arma::vec y_uni(n);
      
      for (int i = 0 ; i < m ; ++i) {
        
        y_uni = series.col(i);
        coef = ARp(y_uni);
        delta[i] = coef["AR1"];
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

//[[Rcpp::export()]]
arma::vec sigma_vec(arma::mat series, int sig2_lag, bool carriero_hack = false) {
  
  int m = series.n_cols, n = series.n_rows;
  
  arma::vec sig2(m);
  arma::vec y_uni(n);
  List AResult;
  NumericVector resid;
  
  for (int i = 0 ; i < m ; ++i) {
    y_uni = series.col(i);
    AResult = ARp(y_uni, sig2_lag);
    resid = AResult["res"];
    
    arma::vec res = resid;
    
    sig2(i) = accu(res % res) / (res.n_elem - sig2_lag - 1);
    
    if (carriero_hack) {
      sig2(i) = accu(res % res) / res.n_elem;
    }
    
  }
  
  return sig2;
  
}



// Creates dummy obs from lambdas
//[[Rcpp::export()]]
List conj_lam2dum(arma::mat series, arma::vec lam, int p, arma::vec delt, 
                  int s2_lag = 1, 
                  arma::mat exo = mat(0, 0), std::string y_bar_type = "initial",
                  bool include_const = true, bool delttypeAR1 = false, bool carriero_hack = false) {
  
  double l_1 = lam[0];
  double l_lag = lam[1];
  double l_sc = lam[2];
  double l_io = lam[3];
  double l_const = lam[4];
  double l_exo = lam[5];
  
  //Rcout << l_1 << ' ' << l_lag << ' ' << l_sc << ' ' << ' ' << l_io << ' ' << l_const << endl;
  
  // m - number of end vars, n - number of obs
  int m = series.n_cols, n = series.n_rows;
  
  // names to vectors have to be added in R!
  
  // add const if needed (as an exogenous variable)
  int d = 0;

  //Rcout << "Check exo empty: " << exo.is_empty() << endl;
  
  if (exo.is_empty()) {
    d = 1 * include_const;
  } else {
    d = exo.n_cols + 1 * include_const;
  }
  
  //Rcout << "d = " << d << endl;
  
  if (include_const) {
    arma::vec intcpt(n) ;
    intcpt.fill(1) ;
    exo = join_rows(intcpt, exo) ;
  }
  
  // k - number of all coefficients
  int k = m * p + d;
  
  // create delta vector from data and description
  arma::vec delta = conj_delta(series, delt, delttypeAR1);
  
  //Rcout << "delta" << endl << delta << endl;
  
  // estimate sigma^2 from AR(p) process (note: Carriero recommends AR(1) )
  arma::vec sigmas_sq;
  sigmas_sq = sigma_vec(series, s2_lag, carriero_hack);
  
  //Rcout << "sigmas_sq" << endl << sigmas_sq << endl;
  
  // get y_bar
  int sc_io_numrows = p;
  if (y_bar_type == "all") {
    sc_io_numrows = n; 
  } else if (y_bar_type != "initial") {
    //Rcout << "y_bar_type not recognised. Set to  'initial' ";
  }
  
  //Rcout << "numrows: " << sc_io_numrows << endl;
  
  arma::vec y_bar = vectorise(mean(series.rows(0, sc_io_numrows - 1), 0));
  
  arma::vec z_bar;
  if (!exo.is_empty()) {
    z_bar = mean(exo.rows(0, sc_io_numrows), 0);
  }
  
  // SC prior setting
  arma::mat Y_sc, X_sc;
  
  if(l_sc != NA_REAL) {
    Y_sc = zeros<mat>(m, m);
    
    Y_sc.diag() = delta % y_bar / l_sc;
    arma::mat exo_dummy(m, d);
    exo_dummy.zeros();
    
    arma::mat temp(1, p);
    temp.ones();
    
    X_sc = join_rows(kron(temp, Y_sc), exo_dummy);
  }
  
  //Rcout << "Y_sc = " << endl << Y_sc << endl;
  //Rcout << "X_sc = " << endl << X_sc << endl;
  
  // io: Initial observation
  arma::mat Y_io, X_io;
  
  if(l_io != NA_REAL) {
    Y_io = ( delta % y_bar ) / l_io;
    for (int i = 0 ; i < p ; ++i) {
      X_io = join_cols(X_io, delta % y_bar / l_io);
    }
  X_io = join_cols(X_io, z_bar / l_io);
  }
  
  //Rcout << "Y_io = " << endl << Y_io << endl;
  //Rcout << "X_io = " << endl << X_io << endl;
  
  // dummy cNIW
  arma::mat y_cniw_block_1, y_cniw_block_2, y_cniw_block_3, y_cniw_block_4;
  
  y_cniw_block_1 = zeros<mat>(sigmas_sq.n_elem , sigmas_sq.n_elem);
  y_cniw_block_1.diag() = (sqrt(sigmas_sq) % delta) / l_1 ;
  
  y_cniw_block_2 = zeros<mat>(m * (p - 1), m);
  
  y_cniw_block_3 = zeros<mat>(sigmas_sq.n_elem , sigmas_sq.n_elem);
  y_cniw_block_3.diag() = sqrt(sigmas_sq);
  y_cniw_block_4 = zeros<mat>(1, m);
    
  
  // carriero hack was ignored
   
  arma::mat Y_cniw = join_cols(y_cniw_block_1, y_cniw_block_2);
  Y_cniw = join_cols(Y_cniw, y_cniw_block_3);
  Y_cniw = join_cols(Y_cniw, y_cniw_block_4);
    
  arma::mat x_cniw_block_1, x_cniw_block_2;
  
  arma::mat temp2 = zeros<mat>(p, p);

  temp2.diag() = pow(linspace<vec>(1, p, p), l_lag);

    x_cniw_block_1 = kron(temp2, y_cniw_block_3) / l_1;
    x_cniw_block_1 = join_rows(x_cniw_block_1, zeros<mat>(m * p, d));
    
    x_cniw_block_2 = zeros<mat>(m, k);
    
    arma::mat x_cniw_block_3 = zeros<mat>(1, m * p);
    
    if (include_const) {
      arma::vec temp3 = ones<vec>(include_const);
      temp3 = temp3 / l_const;
      x_cniw_block_3 = join_rows(x_cniw_block_3, temp3);
    }
    
    //Rcout << "pass1" << endl;
    
    if (d > 1) {
      arma::vec temp4 = ones<vec>(include_const);
      temp4 = temp4 / l_exo;
      x_cniw_block_3 = join_cols(x_cniw_block_3, temp4);
    }
    
    //Rcout << "x_cniw_block_1" << endl << x_cniw_block_1 << endl;
    //Rcout << "x_cniw_block_2" << endl << x_cniw_block_2 << endl;
    //Rcout << "x_cniw_block_3" << endl << x_cniw_block_3 << endl;
    
    //Rcout << "y_cniw_block_1" << endl << y_cniw_block_1 << endl;
    //Rcout << "y_cniw_block_2" << endl << y_cniw_block_2 << endl;
    //Rcout << "y_cniw_block_3" << endl << y_cniw_block_3 << endl;
    //Rcout << "y_cniw_block_4" << endl << y_cniw_block_4 << endl;
    
    arma::mat X_cniw = join_cols(x_cniw_block_1, x_cniw_block_2);
    X_cniw = join_cols(X_cniw, x_cniw_block_3);
    
    ////Rcout << "Y_cniw" << endl << Y_cniw << endl;
    ////Rcout << "X_cniw" << endl << X_cniw << endl;
    
    // Create X_plus and Y_plus
    
    arma::mat X_plus = join_cols(X_cniw, X_sc);
    X_plus = join_cols(X_plus, X_io.t());
    
    arma::mat Y_plus = join_cols(Y_cniw, Y_sc);
    Y_plus = join_cols(Y_plus, Y_io.t());
  
  return List::create(Named("X_sc") = X_sc,
                      Named("Y_sc") = Y_sc,
                      Named("X_io") = X_io,
                      Named("Y_io") = Y_io,
                      Named("X_cniw") = X_cniw,
                      Named("Y_cniw") = Y_cniw,
                      Named("X_plus") = X_plus,
                      Named("Y_plus") = Y_plus,
                      Named("sigmas_sq") = sigmas_sq);
}

//[[Rcpp::export()]]
arma::vec conj_dum2hyp() {
  return 0;
}

//[[Rcpp::export()]]
List conj_sim() {
  return 0;
}

//[[Rcpp::export()]]
List BVAR_cniw_setup (arma::mat series, arma::vec lam, int p, arma::vec delt, int v_prior, 
                      int s2_lag = 1, 
                      arma::mat exo = mat(0, 0), std::string y_bar_type = "initial",
                      bool include_const = true, bool delttypeAR1 = false, bool carriero_hack = false) {
  
  arma::mat exo1 = mat(0, 0);
  Rcout << "check1" << exo1 << endl;
  
  List data = prepData(series, p, include_const);
  
  arma::mat X = data["X"];
  arma::mat Y = data["Y"];
  int m = data["k"];
  
  List dum = conj_lam2dum(series, lam, p, delt, s2_lag, exo, 
                          y_bar_type, include_const, delttypeAR1, carriero_hack);
  
  if (v_prior == -1) {
    v_prior = m + 2;
  }
  
  return List::create(Named("X") = X,
                      Named("Y") = Y,
                      Named("X_plus") = dum["X_plus"],
                      Named("Y_plus") = dum["Y_plus"],
                      Named("v_prior") = v_prior,
                      Named("p") = p,
                      Named("Constant") = include_const);
  
}

List BVAR_conj_est() {
  return 0;
}