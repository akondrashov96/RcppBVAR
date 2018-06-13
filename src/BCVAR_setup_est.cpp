#include <RcppArmadillo.h>
//[[Rcpp::plugins(cpp11)]]
//[[Rcpp::depends(RcppArmadillo)]]
//#include "include/prepData.h"

using namespace Rcpp;
using namespace arma;

List makeData(arma::mat series, int p, bool include_const = true) {
  
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

arma::mat normmat(arma::mat X){
  arma::mat Y = pow(X, 2);
  arma::vec n = sqrt(sum(Y, 0).t()) + std::numeric_limits<double>::epsilon();
  Y = X.each_row() / n.t();
  return Y;
}

//[[Rcpp::export()]]
arma::mat RPmat(int m_l, int K) {
  
  std::random_device                  rand_dev;
  std::mt19937                        generator(rand_dev());
  std::uniform_real_distribution<double>  distr(0.1, 0.9);
  
  double phi = distr(generator);
  
  // matrix is generated as in (Guhaniyogi, Dunson 2015)
  
  arma::vec prob = {phi*phi, 2*(1 - phi)*phi, (1 - phi)*(1 - phi)};
  arma::vec val = {-1/sqrt(phi), 0, 1/sqrt(phi)};
  arma::vec cprob = cumsum(prob);
  
  arma::mat Phi = randu<mat>(m_l, K);
  
  Phi.elem(find(Phi < cprob[0])).fill(val[0]);
  Phi.elem(find(Phi >= cprob[0] && Phi < cprob[1])).fill(val[1]);
  Phi.elem(find(Phi >= cprob[1])).fill(val[2]);
  
  if (m_l > 1) {
    Phi = normmat(Phi);
  }
  
  return Phi;
}


List BCVAR_hyp2dum(arma::mat Omega, arma::mat S, arma::mat Phi) {
  
  arma::mat X_block_1 = chol(Omega).i();
  arma::mat Y_block_1 = solve(X_block_1, X_block_1.t() * X_block_1 * Phi);
  
  arma::mat X_block_2 = zeros<mat>(S.n_cols, Omega.n_cols);
  arma::mat Y_block_2 = chol(S);
  
  arma::mat X_plus = join_cols(X_block_1, X_block_2);
  arma::mat Y_plus = join_cols(Y_block_1, Y_block_2);
  
  return List::create(Named("X_plus") = X_plus,
                      Named("Y_plus") = Y_plus);
}

List BCVAR_dum2hyp(arma::mat Y_star, arma::mat X_star, arma::mat Cmat, bool verbose = false) {
  
  arma::mat U;
  arma::vec s;
  arma::mat V;
  arma::mat CX_star;
  
  CX_star = X_star * Cmat.t();
  bool suc = false;
  
  while (!suc) {
    suc = svd(U, s, V, CX_star);
    if (!suc) {
      // if svd fails, remake compression matrix
      if (verbose) {
        Rcout << ("svd failed, compression matrix was generated anew") << endl;
      }
      Cmat = RPmat(Cmat.n_rows, Cmat.n_cols);
      CX_star = X_star * Cmat.t();
    }
  }
  
  arma::mat diag_inv = zeros<mat>(s.n_elem, s.n_elem);
  diag_inv.diag() = 1/s;
  
  arma::mat Omega_root = V * diag_inv * V.t();
  arma::mat Omega = Omega_root * Omega_root;
  
  arma::mat CPhi_star = Omega * (CX_star.t() * Y_star);
  arma::mat E_star = Y_star - CX_star * CPhi_star;
  
  arma::mat S = E_star.t() * E_star;
  
  arma::mat Phi_star_U = Cmat.t() * CPhi_star;
  arma::mat Omegart_U = Cmat.t() * Omega_root * Cmat;
  arma::mat Omega_U = Cmat.t() * Omega * Cmat;
  
  return List::create(Named("Omega_root") = Omega_root,
                      Named("Omega") = Omega,
                      Named("Omegart_U") = Omegart_U,
                      Named("Omega_U") = Omega_U,
                      Named("E_star") = E_star,
                      Named("S") = S,
                      Named("Phi_star") = CPhi_star,
                      Named("UPhi_star") = Phi_star_U,
                      Named("s") = s,
                      Named("U") = U,
                      Named("V") = V);
  
}

List BCVAR_simulate(int v_post, arma::mat Omega_root_post, arma::mat S_post, arma::mat Phi_post, 
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

//[[Rcpp::export()]]
List BCVAR_conj_setup(arma::mat series, int p, int v_prior, 
                      arma::mat Omega, arma::mat S, arma::mat Phi,
                      bool include_const = true) {
  
  List data = makeData(series, p, include_const);
  
  arma::mat X = data["X"];
  arma::mat Y = data["Y"];
  int m = data["k"];
  
  List dum = BCVAR_hyp2dum(Omega, S, Phi);
  
  if (v_prior < 0) {
    v_prior = m + 2;
  }
  
  return List::create(Named("X") = X,
                      Named("Y") = Y,
                      Named("X_plus") = dum["X_plus"],
                      Named("Y_plus") = dum["Y_plus"],
                      Named("v_prior") = v_prior,
                      Named("Omega_prior") = Omega,
                      Named("S_prior") = S,
                      Named("Phi_prior") = Phi,
                      Named("p") = p,
                      Named("Constant") = include_const);
  
}

//[[Rcpp::export()]]
List BCVAR_conj_est(List setup, int keep, std::string type, bool verbose = false, 
                    int n_chains = 1, int n_phi = 10) {
  
  arma::mat X = setup["X"];
  arma::mat Y = setup["Y"];
  arma::mat X_plus = setup["X_plus"];
  arma::mat Y_plus = setup["Y_plus"];
  
  int nT = Y.n_rows;
  
  arma::mat X_star = join_cols(X_plus, X);
  arma::mat Y_star = join_cols(Y_plus, Y);
  
  int K = X.n_cols;
  int M = Y.n_cols;
  //int KM = K * M;
  
  if (verbose) {
    Rcout << "Calculating posterior hyperparameters" << endl;
  }
  
  int m_max = 5 * round(log(K));
  int m_min = 1;
  int Model_N = 0;
  arma::mat Cmat, CX_star;
  arma::vec BIC = zeros<vec>(m_max * n_phi);
  List post_hyp;
  
  arma::cube Phi_cube(K, M, m_max * n_phi), 
             S_cube(M, M, m_max * n_phi), 
             Omegart_cube(K, K, m_max * n_phi),
             Omega_cube(K, K, m_max * n_phi);
  
  for (int m_l = m_min ; m_l < m_max + 1 ; ++m_l) {
    if (verbose) {
      Rcout << "Model: m_l = " << m_l << ", phi = " << 1 << " ";
    }
    for (int j = 0 ; j < n_phi ; ++j) {
      
      if (verbose && ( (j + 1) % 10 == 0) ) {
        Rcout <<  j + 1 << " ";
      }
      
      Cmat = RPmat(m_l, K);
      
      post_hyp = BCVAR_dum2hyp(Y_star, X_star, Cmat, verbose);
      
      BIC[Model_N] = log(det(as<mat>(post_hyp["S"]) / nT)) + 
        (log(nT) / nT) * as<mat>(post_hyp["Phi_star"]).n_elem;
      
      Phi_cube.slice(Model_N) = as<mat>(post_hyp["UPhi_star"]);
      S_cube.slice(Model_N) = as<mat>(post_hyp["S"]);
      Omegart_cube.slice(Model_N) = as<mat>(post_hyp["Omegart_U"]);
      Omega_cube.slice(Model_N) = as<mat>(post_hyp["Omega_U"]);
      
      Model_N++;
    }
    if (verbose) {
      Rcout <<  endl;
    }
  }
  
  if (verbose) {
    Rcout << "Done!" << endl;
  }
  
  arma::vec PSI = BIC - min(BIC);
  arma::vec Post_weights = exp(-0.5*PSI) / sum(exp(-0.5*PSI));
  
  setup["Weights"] = Post_weights;
  setup["BIC"] = BIC;
  
  arma::mat Phi_post = zeros<mat>(K, M);
  arma::mat S_post = zeros<mat>(M, M);
  arma::mat Omegart_post = zeros<mat>(K, K);
  arma::mat Omega_post = zeros<mat>(K, K);
  
  if (type == "all") {
    for (int i = 0 ; i < Post_weights.n_elem ; ++i) {
      Phi_post = Phi_post + Phi_cube.slice(i) * Post_weights(i);
      S_post = S_post + S_cube.slice(i) * Post_weights(i);
      Omegart_post = Omegart_post + Omegart_cube.slice(i) * Post_weights(i);
      Omega_post = Omega_post + Omega_cube.slice(i) * Post_weights(i);
    }
  } else if(type == "max") {
    int ind = Post_weights.index_max();
    Phi_post = Phi_cube.slice(ind);
    S_post = S_cube.slice(ind);
    Omegart_post = Omegart_cube.slice(ind);
    Omega_post = Omega_cube.slice(ind);
  } else {
    Rcout << "No such type option is available. Type is reset to 'all'.";
    for (int i = 0 ; i < Post_weights.n_elem ; ++i) {
      Phi_post = Phi_post + Phi_cube.slice(i) * Post_weights(i);
      S_post = S_post + S_cube.slice(i) * Post_weights(i);
      Omegart_post = Omegart_post + Omegart_cube.slice(i) * Post_weights(i);
      Omega_post = Omega_post + Omega_cube.slice(i) * Post_weights(i);
    }
  }
  
  int v_prior = setup["v_prior"];
  int v_post = v_prior + nT;
  
  if (verbose) {
    Rcout << "Simulating..." << endl;
  }
  if (keep > 0) {
    setup["sample"] = BCVAR_simulate(v_post, Omegart_post, S_post, Phi_post, verbose, keep, n_chains);
  }
  
  setup["v_post"] = v_post;
  setup["S_post"] = S_post;
  setup["Omega_post"] = Omega_post;
  setup["Phi_post"] = Phi_post;
  
  arma::mat U;
  arma::vec s;
  arma::mat V;
  svd(U, s, V, X_star);
  
  arma::mat diag_inv = zeros<mat>(s.n_elem, s.n_elem);
  diag_inv.diag() = 1/s;
  
  arma::mat Omega_root = V * diag_inv * V.t();
  arma::mat Omega = Omega_root * Omega_root;
  
  setup["Omega_post_check"] = Omega;
  
  return setup;

}