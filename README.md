RcppBVAR v.0.1
===============

RcppBVAR is an R package for estimation of BVAR models. To enhance the speed of calculations, the code is written in C++. Note, that the package is in early development. Thanks!

To install the package:
```r
install.packages("devtools")
devtools::install_github("akondrashov96/RcppBVAR")
```

Add package as library:
```r
library(RcppBVAR)
```

Usage example:
```r
freqVAR(series, p = 2)
```
Plans:
- [ ] Add, actually, BVAR models
- [ ] Examples
- [ ] Datasets
