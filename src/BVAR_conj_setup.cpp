#include <RcppArmadillo.h>
//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]
#include "include/prepData.h"

using namespace Rcpp;
using namespace arma;

// calcquantiles 
//[[Rcpp::export()]]
arma::mat CalcQuantiles(const arma::mat& X, const arma::vec& q) {
  arma::mat Y = arma::sort(X, "ascending", 0);
  arma::vec probL = (1 - reverse(q)/100)/2 - std::numeric_limits<double>::epsilon();
  arma::vec probH = (1 + q/100)/2 - std::numeric_limits<double>::epsilon();
  arma::vec prob = join_cols(probL, probH);
  
  arma::mat quantiles;
  for (int i = 0 ; i < prob.n_elem ; ++i) {
    quantiles = join_cols(quantiles, vectorise(Y.row(Y.n_rows*prob[i])).t());
  }
  
  return quantiles;
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
    Rcpp::stop("Length of delta should be equal to 1 or m");
  }  
  
  return delta;
}

//[[Rcpp::export()]]
arma::vec conj_sigma(arma::mat series, int sig2_lag, bool carriero_hack = false) {
  
  int m = series.n_cols, n = series.n_rows;
  
  arma::vec sig2(m);
  arma::vec y_uni(n);
  List AResult;
  
  for (int i = 0 ; i < m ; ++i) {
    y_uni = series.col(i);
    AResult = ARp(y_uni, sig2_lag);
    arma::vec res = as<mat>(AResult["res"]);
    
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
                  Rcpp::Nullable<arma::mat> Z = R_NilValue, std::string y_bar_type = "initial",
                  bool include_const = true, bool delttypeAR1 = false, bool carriero_hack = false) {
  
  double l_1 = lam[0];
  double l_lag = lam[1];
  double l_sc = lam[2];
  double l_io = lam[3];
  double l_const = lam[4];
  double l_exo = lam[5];
  
  // m - number of end vars, n - number of obs
  int m = series.n_cols, n = series.n_rows;
  
  // names to vectors have to be added in R!
  
  // add const if needed (as an exogenous variable)
  int d = 0;
  
  arma::mat exo;
  
  if (Z.isNotNull()) {
    exo = as<mat>(Z);
    d = exo.n_cols + 1 * include_const;
  } else {
    d = 1 * include_const;
  }
  
  if (include_const) {
    arma::vec intcpt(n) ;
    intcpt.fill(1) ;
    exo = join_rows(intcpt, exo) ;
  }
  
  // k - number of all coefficients
  int k = m * p + d;
  
  // create delta vector from data and description
  arma::vec delta = conj_delta(series, delt, delttypeAR1);
  
  // estimate sigma^2 from AR(p) process (note: Carriero recommends AR(1) )
  arma::vec sigmas_sq;
  sigmas_sq = conj_sigma(series, s2_lag, carriero_hack);
  
  // get y_bar
  int sc_io_numrows = p;
  if (y_bar_type == "all") {
    sc_io_numrows = n; 
  } else if (y_bar_type != "initial") {
    Rcout << "y_bar_type not recognised. Set to  'initial' ";
  }
   
  arma::vec y_bar = vectorise(mean(series.rows(0, sc_io_numrows - 1), 0));
  
  arma::vec z_bar;
  if (!exo.is_empty()) {
    z_bar = vectorise(mean(exo.rows(0, sc_io_numrows - 1), 0));
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
  
  // io: Initial observation
  arma::mat Y_io, X_io;
  
  if(l_io != NA_REAL) {
    Y_io = ( delta % y_bar ) / l_io;
    for (int i = 0 ; i < p ; ++i) {
      X_io = join_cols(X_io, delta % y_bar / l_io);
    }
  X_io = join_cols(X_io, z_bar / l_io);
  }

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
    
    if (d > 1) {
      arma::vec temp4 = ones<vec>(include_const);
      temp4 = temp4 / l_exo;
      x_cniw_block_3 = join_cols(x_cniw_block_3, temp4);
    }
    
    arma::mat X_cniw = join_cols(x_cniw_block_1, x_cniw_block_2);
    X_cniw = join_cols(X_cniw, x_cniw_block_3);

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
List conj_dum2hyp(arma::mat Y_star, arma::mat X_star, bool verbose = false) {
  
  arma::mat U;
  arma::vec s;
  arma::mat V;
  svd(U, s, V, X_star);
  
  arma::mat diag_inv = zeros<mat>(s.n_elem, s.n_elem);
  diag_inv.diag() = 1/s;
  
  if (verbose) {
    Rcout << "Calculating Omega..." << endl;
  }
  
  arma::mat Omega_root = V * diag_inv * V.t();
  arma::mat Omega = Omega_root * Omega_root;
  
  if (verbose) {
    Rcout << "Calculating Phi..." << endl;
  }
  
  arma::mat Phi_star = Omega * (X_star.t() * Y_star);
  arma::mat E_star = Y_star - X_star * Phi_star;
  
  if (verbose) {
    Rcout << "Calculating S..." << endl;
  }
  
  arma::mat S = E_star.t() * E_star;
  
  return List::create(Named("Omega_root") = Omega_root,
                      Named("Omega") = Omega,
                      Named("E_star") = E_star,
                      Named("S") = S,
                      Named("Phi_star") = Phi_star,
                      Named("s") = s,
                      Named("U") = U,
                      Named("V") = V);
}

//[[Rcpp::export()]]
List conj_simulate(int v_post, arma::mat Omega_root_post, arma::mat S_post, arma::mat Phi_post, 
                   bool verbose = false, int keep = 10, int chains = 1) {
  
  int k = Phi_post.n_rows, m = Phi_post.n_cols;
  
  int thres = keep / 10;
  if (k > 80) {
    Rcout << "The model contains over 80 parameters. " << endl <<
      "Verbose is set to TRUE and messages are shown more frequently to assess progress" << endl;
    thres = 50;
    verbose = true;
  }
  
  List sims;
  arma::mat answer = zeros<mat>(keep, m * k + m * m);
  arma::mat Sigma, V, Phi;
  arma::vec Phi_vec, Sigma_vec, plug_in;
  
  for (int chain = 0 ; chain < chains ; ++chain) {
    if (verbose) {
      Rcout << "===============================================" << endl;
      Rcout << "Chain: " << chain + 1 << " out of " << chains << endl;
      Rcout << "===============================================" << endl;
    }
    for (int i = 0 ; i < keep ; ++i ) {
      if (verbose && (i % thres == 0) ) {
        Rcout << "Iteration " << i + 1 << " out of " << keep << endl;
      }
      
      if (verbose && (i % thres == 0) ) {
        Rcout << i << ": Calculating Sigma... " << "\t";
      }
      Sigma = iwishrnd(S_post, v_post);
      V = randn<mat>(k, m);
      
      if (verbose && (i % thres == 0) ) {
        Rcout << "Calculating Phi... " << endl;
      }
      Phi = Phi_post + Omega_root_post * V * chol(Sigma);
      
      Phi_vec = vectorise(Phi);
      Sigma_vec = vectorise(Sigma);
      plug_in = join_cols(Phi_vec, Sigma_vec);
      
      answer.row(i) = plug_in.t();
      
    }
    if (verbose) {
      Rcout << endl;
    }
    sims["chain" + std::to_string(chain + 1)] = answer;
    answer = zeros<mat>(keep, m * k + m * m);
  }

  if (verbose) {
    Rcout << "Done!" << endl;
  }
  return sims;
}

//[[Rcpp::export(bvar_setup)]]
List BVAR_cniw_setup (arma::mat series, arma::vec lam, int p, arma::vec delt, int v_prior, 
                      int s2_lag = 1,
                      Rcpp::Nullable<arma::mat> Z = R_NilValue, std::string y_bar_type = "initial",
                      bool include_const = true, bool delttypeAR1 = false, bool carriero_hack = false) {
  
  List data = prepData(series, p, include_const);
  
  arma::mat X = data["X"];
  arma::mat Y = data["Y"];
  int m = data["k"];
  
  List dum = conj_lam2dum(series, lam, p, delt, s2_lag, Z, 
                          y_bar_type, include_const, delttypeAR1, carriero_hack);
  
  // pretty much unneeded, but I'll leave it just in case
  if (v_prior < 0) {
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

//[[Rcpp::export(bvar_est)]]
List BVAR_cniw_est(List setup, int keep, bool verbose = false, int n_chains = 1) {
  
  arma::mat X = setup["X"];
  arma::mat Y = setup["Y"];
  arma::mat X_plus = setup["X_plus"];
  arma::mat Y_plus = setup["Y_plus"];
  
  int nT = Y.n_rows;
  
  arma::mat X_star = join_cols(X_plus, X);
  arma::mat Y_star = join_cols(Y_plus, Y);
  
  if (verbose) {
    Rcout << "Calculating posterior hyperparameters" << endl;
  }
  
  List post = conj_dum2hyp(Y_star, X_star);
  int v_prior = setup["v_prior"];
  int v_post = v_prior + nT;
  
  if (verbose) {
    Rcout << "Simulating..." << endl;
  }
  if (keep > 0) {
    setup["sample"] = conj_simulate(v_post, post["Omega_root"], post["S"], post["Phi_star"], 
                              verbose, keep, n_chains);
  }
  
  setup["v_post"] = v_post;
  setup["S_post"] = post["S"];
  setup["Omega_post"] = post["Omega"];
  setup["Phi_post"] = post["Phi_star"];
  
  
  return setup;
}

//[[Rcpp::export(bvar_fcst)]]
List BVAR_cniw_forecast(List model, arma::mat series, Rcpp::Nullable<arma::mat> Z_f = R_NilValue,
                        int h = 1, bool out_of_sample = true, 
                        std::string type = "prediction", 
                        Rcpp::Nullable<List> level_ = R_NilValue,
                        Rcpp::Nullable<StringVector> include_ = R_NilValue,
                        bool fast_forecast = false, bool verbose = false) {
  
  // import data from model
  // assume that series data is ALWAYS present - no need to restore it
  
  if (verbose) {
    Rcout << "R passed the data to C++. Commencing." << endl;
  }
  
  StringVector include;
  if (include_.isNotNull()) {
    include = include_.get();
  } else {
    include = {"mean", "median", "sd", "interval", "raw"};
  }
  
  NumericVector level;
  if (level_.isNotNull()) {
    level = level_.get();
  } else {
    level = {80, 95};
  }
  
  if (verbose) {
    Rcout << "Copying parameters..." << endl;
  }
  
  bool constant = model["Constant"];
  arma::mat Y = as<mat>(model["Y"]);
  arma::mat X = as<mat>(model["X"]);
  arma::mat X_plus = as<mat>(model["X_plus"]);
  
  int nT = Y.n_rows, p = model["p"], 
      k = X.n_cols, m = Y.n_cols; 
  
  if (verbose) {
    Rcout << "Checking data for errors..." << endl;
  }
  
  // check for errors in data
  arma::mat Zf;
  if (Z_f.isNotNull()){
    Zf = as<mat>(Z_f);
    if (Zf.n_rows != h) {
      Rcpp::stop("ERROR: Number of rows in Z_f must be equal to h");
    }
  }
  // add constant to exo variables
  if (constant) {
    arma::vec intcpt(h) ;
    intcpt.fill(1) ;
    Zf = join_rows(Zf, intcpt);
  }
  
  if (series.n_rows < p) {
    Rcpp::stop("ERROR: insufficient observations in provided series");
  }
  
  // set initial number of iterations
  int keep = 0;
  
  // take the needed initial number of observations for forecasting
  if (out_of_sample) { // last p ones for out of sample
    series = series.rows(series.n_rows - 1 - p, series.n_rows - 1);
  } else { // first p ones for in-sample
    series = series.rows(0, p);
    h = nT;
  }
  
  // if sample is available, set keep to n_rows
  arma::mat fullsamp;
  if (model.containsElementNamed("sample")){
    List samples = model["sample"];
    fullsamp = as<mat>(samples[0]);
    for (int i = 1 ; i < samples.length() ; ++i) {
      fullsamp = join_cols(fullsamp, as<mat>(samples[i]));
    }
    keep = fullsamp.n_rows;
  }
  
  // if no sample and fast_forcast is not desired, force fast forecast
  if ( (keep == 0) && (!fast_forecast)) {
    Rcout << "The model contains no simulations. Forecast is set to fast_forecast = TRUE" << endl;
    fast_forecast = true;
  }

  // settings for fast forecast
  if (fast_forecast) {
    include = "mean";
    keep = 1;
    
    type = "credible"; // to avoid epsilon simulation
  }
  
  if (verbose) {
    Rcout << "Initialisation of forecasting variables..." << endl;
  }
  
  // initialise matrix for forecasted values
  arma::mat forecast_raw = zeros<mat>(keep, m * h);
  arma::vec e_t = zeros<vec>(m), x_t = zeros<vec>(k), y_hat_temp, y_hat;
  arma::mat Phi(k, m), Sigma(m, m), tPhi, R, Y_init;
  int nobs = series.n_rows;
  
  if (verbose) {
    Rcout << "Forecasting... " << endl;
  }

  // note: fast forecast will go only once, as keep = 1
  // normal forecast will go n_rows times
  for (int i = 0 ; i < keep ; ++i) {
    // forecast h steps for given sample of Phi
    
    if (verbose && (i % 1000 == 0) ) {
        Rcout << "Iteration " << i + 1 << " out of " << keep << endl;
      }
    
    if (fast_forecast) {
      // use posterior exp value of Phi provided by the model
      Phi = as<mat>(model["Phi_post"]); 
    } else {
      // now we use the mcmc sample
      Phi = reshape((fullsamp.row(i)).cols(0, k * m - 1), k, m);
      Sigma = reshape((fullsamp.row(i)).cols(k * m, m * k + m * m - 1), m, m);
      
      vec eigval;
      mat eigvec;
      
      eig_sym(eigval, eigvec, Sigma);
      
      for (int val = 0 ; val < eigval.n_elem ; ++val) {
        if (eigval[val] <= -sqrt(std::numeric_limits<double>::epsilon() * std::abs(eigval[val]))) {
          Rcpp::warning("Omega_post is not positive definite numerically");
        }
      }
      
      R = (eigvec * (eigvec.each_col() % sqrt(eigval))).t();
    
    }
      
    tPhi = Phi.t();
     
    for (int j = 0 ; j < h ; ++j) {
      if (out_of_sample) {
        // get x_t for out-of-sample forecasts
          if (j >= 1) {
            Y_init = join_cols(y_hat_temp.t(), Y_init.rows(0, p - 2));
          } else {
            Y_init = series.rows(nobs - p, nobs - 1);
            Y_init = reverse(Y_init, 0);
          }
          
          x_t = join_cols(vectorise(Y_init.t()), Zf.row(j));
        } else {
          // get x_t for in-sample forecasts

          x_t = X.row(j).t();

        }
        

        if (type == "prediction") {
            e_t = R * randn<vec>(m);
        }
        
        y_hat_temp = tPhi * x_t + e_t;
        y_hat = join_cols(y_hat, y_hat_temp);

    }
    
    //Rcout << "forecast_raw" << endl << forecast_raw << endl;
    forecast_raw.row(i) = y_hat.t();
    y_hat.reset();
    
  }
  
  arma::vec avg, med, stdev;
  arma::mat qnt;
  
  // get means
  if (std::find(include.begin(), include.end(), "mean") != include.end()) {
    avg = vectorise(mean(forecast_raw, 0));
  }
  
  // get median
  if (std::find(include.begin(), include.end(), "median") != include.end()) {
    med = vectorise(median(forecast_raw, 0));
  }
  
  // get stdev
  if (std::find(include.begin(), include.end(), "sd") != include.end()) {
    stdev = vectorise(stddev(forecast_raw, 0));
  }
  
  // get quantiles
  if (std::find(include.begin(), include.end(), "interval") != include.end()) {
    qnt = CalcQuantiles(forecast_raw, level);
  }
    
  if (verbose) {
    Rcout << "Done!" << endl;
  }
  
  return List::create(Named("raw") = forecast_raw,
                      Named("mean") = avg,
                      Named("median") = med,
                      Named("sd") = stdev,
                      Named("quantiles") = qnt);
}
