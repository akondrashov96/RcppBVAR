% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Functions.R
\name{BVAR_conj_estimate}
\alias{BVAR_conj_estimate}
\title{Estimate Conjugate Normal Inverse Wishart BVAR model}
\usage{
BVAR_conj_estimate(mod_setup, keep = 2000, n_chains = 1, verbose = FALSE)
}
\arguments{
\item{keep}{Number of simulations. If set to 0, no simulation sample is created, only posterior parameters.}

\item{n_chains}{Number of chains (for convergence)}

\item{verbose}{Whether to include messages.}

\item{setup}{The setup provided by \code{BVAR_conj_setup}.}
}
\value{
A list with: setup parameters (priors), posterior parameters (S, Phi, Omega and v), 
and sample, if requested.
}
\description{
Obtain posterior parameters and a sample of Phi and sigma matrix
}
\details{
The function takes prior parameters (from \code{BVAR_conj_setup}) and estimates the model.
}
