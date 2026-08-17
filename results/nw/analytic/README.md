# Analytic NW Mechanism Experiment

The model is `X ~ Unif[-1,1]`, `Y = beta + b X + A X^2 + epsilon`, with a perfect predictor equal to the conditional mean. The target is `theta(0)=beta`. The configuration is `beta=0`, `A=4`, `sigma=1`, `n=200`, and `N=5000`.

The NW calculations use the analytic uniform localization weight. As `h` decreases, NW localization bias falls, but the local variation explained by the fixed-covariate predictor also falls. Consequently, the relative prediction-powered variance reduction approaches zero.

The log-log slope uses the smallest 40% of the predeclared bandwidth grid. Estimated slopes are b=0: 3.999, b=2: 1.995, matching the predicted orders 2 and 4.

The horizontal signed-weight references use `w*(x)=9/4-15x^2/4`, the exact signed-balance representer only in `span{1,x,x^2}`. They are finite-dimensional references, not claims that a general RKHS localization estimator recovers `w*`.

`normal_approx_coverage` is a normal-approximation calculation and is not exact finite-sample coverage.
