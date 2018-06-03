#' RcppBVAR
#'
#' @name RcppBVAR
#' @docType package
#' @author Artem Kondrashov
NULL

#' Conjugate Normal Inverse Wishart BVAR Forecasting
#'
#' Forecasts with CNIW BVAR. The function takes \code{BVAR_cniw_est()} result as model.
#'
#' The functions uses MCMC approach for prediction and generates a large sample 
#' (depends on size of Phi draws in the estimated model). Mean, median, standard seviation and 
#' quantiles are supplied automatically. The computations are done in C++ (\code{bvar_fcst()} function).
#' The R function acts as a shell providing inputs and making a list of output dataframe and raw 
#' forecast table, if requested by the user.
#'
#' @param model A model estimated by \code{BVAR_cniw_est()} function.
#' 
#' @param src_series The source series used in estimation and forecasting. 
#' Required for forecasting.
#' 
#' @param Z_f Forecasted exogenous variables.
#' 
#' @param h Forecast horizon. Default is \code{1}.
#' 
#' @param out_of_sample Whether the forecast is done in-sample or out-of-sample. Default is \code{TRUE}.
#' 
#' @param type If set to \code{"prediction"} (default value), then uncertainty 
#' about future shocks is incorporated (by adding random errors). If set to \code{"credible"}, 
#' only parameter uncertainty is considered.
#' 
#' @param level Confidence levels for prediction intervals. Can be given as a vector.
#' 
#' @param include What type of descriptive statistics on raw forecast sample to include in the summary.
#' By default, mean, median, standard deviation, quantiles and raw data is included (the latter is suplied 
#' in a separate list element)
#' 
#' @param fast_forecast Default is \code{FALSE}. If \code{TRUE}, the MCMC forecasting is skipped and 
#' only posterior Phi matrix is used.
#' 
#' @param verbose Default is \code{FALSE}. If set to \code{TRUE} internal messages will be displayed.
#' 
#' @return A list containing: a dataframe with summary, raw forecasts sample (if \code{raw} is in \code{include}).
#' @export
#' @examples
#' ### Dummy Code: Do not run ###
#' data(series)
#' setupCpp <- BVAR_cniw_setup(train, lam = c(0.2, 1, 1, 1, 100, 100), p = 4, s2_lag = 4, v_prior = -1, delt = 1)
#' estCpp <- BVAR_cniw_est(setup = setupCpp, keep = 2000, verbose = TRUE, n_chains = 4)
#' fcast <- BVAR_conj_forecast(model = estCpp, src_series = train, h = nrow(test), out_of_sample = FALSE, 
#'                             level = c(80, 95), verbose = TRUE)
                            
BVAR_conj_forecast <- function(model, src_series, Z_f = NULL, h = 1, out_of_sample = TRUE, 
                               type = c("prediction", "credible"), level = c(80, 95), 
                               include = c("mean", "median", "sd", "interval", "raw"), 
                               fast_forecast = FALSE, verbose = FALSE) {
  
  type <- match.arg(type)
  eqNames <- colnames(src_series)
  
  if (verbose) {
    print("Passing to C++...")
  }
  
  result <- bvar_fcst(model = model, series = src_series, Z_f = Z_f, 
                       h = h, out_of_sample = out_of_sample, type = type, 
                       level_ = level, include_ = include, 
                       fast_forecast = fast_forecast, verbose = verbose)
  
  if (!out_of_sample) {
    h = nrow(model$Y)
  }
  
  if (fast_forecast) {
    df_forecast <- data.frame(series = rep(eqNames, h),
                              h = rep(1:h, each = length(eqNames)),
                              mean = result$mean)
  } else {
    df_forecast <- data.frame(series = rep(eqNames, h),
                              h = rep(1:h, each = length(eqNames)),
                              mean = result$mean,
                              median = result$median,
                              sd = result$sd)
    
    qt <- t(result$quantiles)
    colnames(qt) <- paste(formatC(c((100 - rev(level))/2, (100 + level)/2), 
                                  format = "f", digits = 1), "%", sep = "")
    df_forecast <- cbind(df_forecast, qt)
  }
  
  output <- list(summary = df_forecast)
  
  if ("raw" %in% include) {
    output[["raw"]] = result$raw
    colnames(output[["raw"]]) <- paste(eqNames, ", h = ", 
                                       rep(1:h, each = length(eqNames)), 
                                       sep = "")
  }
  
  return(output)
}



#' Create a setup for Conjugate Normal Inverse Wishart BVAR model
#'
#' Assemble all prior and hyperparameter data from provided lambdas.
#'
#' The function takes hyperparameter data and constructs a list of data suitable for 
#' \code{BVAR_conj_estimate()} function. Calculations are done by a C++ function (callable to R as \code{bvar_setup}).
#' 
#'
#' @param series Time series for estimation.
#' 
#' @param lambda A set of lambdas (default: \code{l_1 = 0.2, l_lag = 1, l_sc = 1, l_io = 1, l_const = 100, l_exo = 100}).
#' 
#' @param p Number of lags.
#' 
#' @param delta A vector [m x 1]/ scalar / \code{"AR1"}. Used for Phi prior. If \code{delta} is set to \code{"AR1"}
#' then AR(1) model is used to calculate delta.
#' 
#' @param v_prior Prior value of hyperparameter nu. By default \code{m + 2}.
#' 
#' @param s2_lag Number of lags in AR() model used to estimate sig^2. By default is set to \code{p}.
#' 
#' @param Z_mat Matrix of exogenous variables
#' 
#' @param y_bar_type \code{"initial"} or \code{"all"}. Defines how y_bar is calculated for sum-of-coefficients and 
#' initial observation dimmy variables. Default is \code{"initial"}.
#' 
#' @param include_const Default \code{TRUE}. Whether to include the constant in the model.
#' 
#' @param carriero_hack Logical, if \code{TRUE} sigma^2 will be estimated using biased estimator
#' and supposed error with no square roots in dummy observations will be reproduced.
#' 
#' @return A list containing: Y and X matrices, X_plus and Y_plus matrices, v_prior, p and whether constant is 
#' included.
#' 
#' @export
#' @example
#' ### Dummy code: do not run ###
#' data(series)
#' setup <- BVAR_conj_setup(series, p = 4, delta = 1, include_const = TRUE, y_bar_type = "initial")

BVAR_conj_setup <- function(series, lambda = c(0.2, 1, 1, 1, 100, 100), p = 1, delta = 1, v_prior = NULL,
                            s2_lag = NULL, Z_mat = NULL, y_bar_type = c("initial", "all"),
                            include_const = TRUE, carriero_hack = FALSE) {
  
  # reset s2_lag to p if unspecified
  if (is.null(s2_lag)) {
    s2_lag <- p
  }
  
  # to pass it in C++ function correctly
  y_bar_type <- match.arg(y_bar_type)
  
  # correct series and exogenous vars (just in case)
  series <- as.matrix(series)
  if (!is.null(Z_mat)) {
    Z_mat <- as.matrix(Z_mat)
  }
  
  if (is.null(v_prior)) {
    v_prior <- ncol(series) + 2
  }
  
  # correct delta
  delttypeAR1 = FALSE
  if (delta[1] == "AR1") {
    delttypeAR1 <- TRUE
    delta = 1
  }
  
  output <- bvar_setup(series = series, lam = lambda, p = p, 
                       delt = delta, v_prior = v_prior, s2_lag = s2_lag, 
                       Z = Z_mat, y_bar_type = y_bar_type, 
                       include_const = include_const, 
                       delttypeAR1 = delttypeAR1, carriero_hack = carriero_hack)
  
  return(output)
}

#' Estimate Conjugate Normal Inverse Wishart BVAR model
#'
#' Obtain posterior parameters and a sample of Phi and sigma matrix
#' 
#' The function takes prior parameters (from \code{BVAR_conj_setup}) and estimates the model.
#' 
#' @param setup The setup provided by \code{BVAR_conj_setup}.
#' 
#' @param keep Number of simulations. If set to 0, no simulation sample is created, only posterior parameters.
#' 
#' @param n_chains Number of chains (for convergence)
#' 
#' @param verbose Whether to include messages.
#' 
#' @return A list with: setup parameters (priors), posterior parameters (S, Phi, Omega and v), 
#' and sample, if requested.
#' 
#' @export
#' @example 
#' ### Dummy code: Do not run ###
#' data(series)
#' setup <- BVAR_conj_setup(series, p = 4, delta = 1, include_const = TRUE, y_bar_type = "initial")
#' estim <- BVAR_conj_estimate(setup, keep = 2000, n_chains = 4, verbose = FALSE)
#'

BVAR_conj_estimate <- function(mod_setup, keep = 2000, n_chains = 1, verbose = FALSE) {
  
  output <- bvar_est(setup = mod_setup, keep = keep, n_chains = n_chains, verbose = verbose)
  
  return(output)
}
