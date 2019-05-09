# Talk on Variational Inference and scalable Bayesian Methods

Lecture notes containing discussion of computational complexity and introduction to VI. Presents CAVI and BBVI examples as well as notes on more current research and future directions. Written as an rmarkdown, compiles with `R -e "rmarkdown::render('presentation.Rmd')"`.

`logistic.jl` contains a Metropolis Hastings sampler for logistic regression and a black box VI implementation. This should be run before compiling the lecture notes as it creates a results csv. Note: the `GradDescent` package must be installed using `add https://github.com/jacobcvt12/GradDescent.jl` since the release version is not 1.0+ compatible.
