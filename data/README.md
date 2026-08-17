# Data

- `census_income/census_income.npz` contains the processed covariates, outcome, and auxiliary predictions used by the Income experiment. Income is measured in units of 10,000 dollars.
- `blogfeedback/blogfeedback.zip` contains the BlogFeedback training data used by the experiment script. The response is transformed with `log1p`.

The experiment scripts resolve these paths relative to the repository root. The BlogFeedback script trains its auxiliary predictor on a split disjoint from the PPCI sampling pool.
