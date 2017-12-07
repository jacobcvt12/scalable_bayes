# load packages
using Distributions, ForwardDiff
using StatsFuns
using GradDescent

# data size
N = 10000  # number of observations
D = 20     # number of covariates including intercept

# generate data
srand(1)                  # insure reproducibility
X = rand(Normal(), N, D)  # generate covariates
X[:, 1] = 1               # make first column intercept
b = rand(Normal(), D)     # coefficients
θ = logistic.(X * b)      # simulate dependent variables
y = rand.(Bernoulli.(θ))  # simulate dependent variables

# fit bayesian logistic regression with Normal(0, 1) priors 
# using variational inference with reparam trick
# (Kingma and Welling 2014)

# estimate of evidence lower bound
function ℒ(y, X, prior, μ, σ)
    # generate L=1 sample from q(z|μ, σ)
    ϵ = rand(Normal(), D)
    z = μ + ϵ .* σ

    # calculate log prior
    log_prior = logpdf.(prior, z)

    # calculate log likelihood
    θ_z = logistic.(X * z)
    log_lik = sum(logpdf.(Bernoulli.(θ_z), y))


    # calculate log joint density
    log_joint = log_prior + log_lik

    # calculate log variational density 
    # (often referred to as entropy)
    entropy = logpdf.(Normal.(μ, σ), z)

    # estimate ELBO with L=1 sample
    f = log_joint - entropy

    return f
end

# practical consideration
# nonnegative parameters must be truncated
# in the interest of numerical stability
function truncate!(λ, D)
    # truncate variational standard deviation
    for d in 1:D
        λ[d, 2] = λ[d, 2] < invsoftplus(1e-5) ? 
                  invsoftplus(1e-5) : λ[d, 2]
        λ[d, 2] = λ[d, 2] > softplus(log(prevfloat(Inf))) ? 
                  invsoftplus(log(prevfloat(Inf))) : λ[d, 2]
    end
end

# stochastic variational inference implementation of
# logistic regression
function svi(y, X, prior, opt; 
             tol=0.005, maxiter=1000, elboiter=10, info=25)
    N = length(y)
    D = size(X, 2)

    # initialize variational parameters
    λ = rand(Normal(), D, 2)
    truncate!(λ, D)

    mean_δ = 1.0
    i = 1

    while mean_δ > tol && i < maxiter
        # pull out mean and transformed SD
        # for variational densities
        # note: each regression parameter
        # has the posterior approximated by N(μ[i], σ[i])
        μ = λ[:, 1]
        σ = softplus.(λ[:, 2])

        # estimate gradient of ELBO using current values of ϕ
        # note the use of automatic differentiation here
        g = ForwardDiff.jacobian(ϕ -> ℒ(y, X, prior, ϕ[1:D], 
                                        ϕ[(D+1):2D]), 
                                 vcat(μ, σ))
        ∇ℒ = hcat(diag(g[1:D, 1:D]),
                  diag(g[1:D, (D+1):2D]))

        # calculate size of gradient ascent step
        δ = update(opt, ∇ℒ)
        λ += δ

        # necessary truncation
        truncate!(λ, D)

        # calculate the change of parameters
        mean_δ = mean(δ .^ 2)

        # status update
        if i % info == 0
            println(i, " ", mean_δ)
        end
        
        i += 1
    end

    # calculate elbo
    μ = λ[:, 1]
    σ = softplus.(λ[:, 2])
    elbo = 0

    for i in 1:elboiter
        elbo += mean(ℒ(y, X, prior, μ, σ))
    end

    elbo = elbo / elboiter

    return λ, elbo
end

# metropolis hastings implementation of non-conjugate 
# logistic regression
function mh(y, X, prior; S=2000, delta=0.1, info=25)
    N = length(y)
    D = size(X, 2)

    # allocate MCMC chain
    θ = zeros(S, D)

    # initialize samples
    θ[1, :] = rand(prior, D)

    accept = zeros(D)

    # loop over chain
    for s in 2:S
        # status update
        if s % info == 0
            println(s)
        end

        # sweep over variables
        θ_j = copy(θ[s-1, :])

        for d in 1:D
            θ_star = copy(θ_j)

            # draw from metropolis proposal
            θ_star[d] = rand(Normal(θ_j[d], delta), 1)[1]

            # calculate regression probabilities
            # for current and proposed value
            μ = logistic.(X * θ_j)
            μ_star = logistic.(X * θ_star)

            # calculate log likelihood for current and
            # proposed value
            log_lik = sum(logpdf.(Bernoulli.(μ), y))
            log_lik_star = sum(logpdf.(Bernoulli.(μ_star), y))

            # calculate log prior for current and
            # proposed value
            log_prior = logpdf(prior, θ_j[d])
            log_prior_star = logpdf(prior, θ_star[d])

            # note that we don't need MH densities
            # since proposal distribution is symmetric

            # calculate acceptance probability
            R = log_lik_star + log_prior_star - 
                (log_lik + log_prior)

            # determine if proposal should be 
            # accepted or rejected
            U = rand()
            if log(U) < R
                θ[s, d] = θ_star[d]
                θ_j[d] = θ_star[d]
                accept[d] += 1
            else
                θ[s, d] = θ_j[d]
                θ_star[d] = θ_j[d]
            end 
        end
    end

    # print acceptance ratio for tuning proposal distribution
    print(accept / S)

    return θ

end

# fit multiple VI models and compare
fits = 3

srand(428)

λ = zeros(D, 2, fits)
elbo = zeros(fits)

tic()
for fit in 1:fits
    # prior on coefficents
    prior = Normal(0.0, 1.0)

    # initialize optimizer
    opt = Adam(α=1.0)

    # fit model 
    λ[:, :, fit], elbo[fit] = svi(y, X, prior, opt)
end
toc()

# model with highest ELBO should be selected
best = findmax(elbo)[2]
λ = λ[:, :, best]

# now fit MH model
tic()
θ = mh(y, X, Normal(), delta=0.15, info=100)
toc()

using Gadfly, DataFrames

lb = quantile.(Normal.(λ[:, 1], softplus.(λ[:, 2])), 0.025)
ub = quantile.(Normal.(λ[:, 1], softplus.(λ[:, 2])), 0.975)

results = DataFrame(variable=1:D,
                    truth=b,
                    vi=λ[:, 1],
                    mh=mean(θ[1001:2000, :], [1])[1, :],
                    lb=lb,
                    ub=ub)
results = stack(results, [:truth, :vi, :mh])

plot(results, x=:variable_1, y=:value, ymin=:lb, ymax=:ub, color=:variable, Geom.point, Geom.errorbar)
