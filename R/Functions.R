#' RcppBVAR
#'
#' @name RcppBVAR
#' @docType package
#' @author Artem Kondrashov
NULL

#' Conjugate Normal Inverse Wishart BVAR Forecasting
#'
#' Forecasts with CNIW BVAR. The function takes BVAR_cniw_est() result as model.
#'
#' The functions uses MCMC approach for prediction and generates a large sample 
#' (depends on size of Phi draws in the estimated model). Mean, median, standard seviation and 
#' quantiles are supplied automatically. The computations are done in C++ (hidden .bvar_fcst() function).
#' The R function acts as a shell providing inputs and making a list of output dataframe and raw 
#' forecast table, if requested by the user.
#'
#' @param model A model estimated by BVAR_cniw_est() function.
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
#' @param type If set to \preformatted{"prediction"} (default value), then uncertainty 
#' about future shocks is incorporated (by adding random errors). If set to \preformatted{"credible"}, 
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
#' @return A list containing: a dataframe with summary, raw forecasts sample (if raw is in \code{include}).
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
  eqNames <- colnames(series)
  result <- .bvar_fcst(model = model, series = src_series, Z_f = Z_f, 
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