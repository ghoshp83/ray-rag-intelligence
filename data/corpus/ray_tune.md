# Ray Tune

Ray Tune is the hyperparameter optimisation library in Ray. It runs many
training trials in parallel across a cluster and uses search algorithms and
schedulers to find good configurations efficiently. Tune is framework-agnostic
and integrates with Ray Train, so a distributed training job can itself be the
unit that Tune optimises.

A tuning job is defined by a search space, which describes the hyperparameters
and the ranges or distributions to sample, and by a search algorithm that decides
which configurations to try next. Random search and grid search are available,
along with more sample-efficient algorithms such as Bayesian optimisation.

Schedulers add a second layer of efficiency by stopping unpromising trials early
so that compute is concentrated on the configurations that look most likely to
succeed. The Asynchronous Successive Halving Algorithm, known as ASHA, is a
popular scheduler: it runs many trials for a short budget, keeps the best
performers, and progressively allocates more resources to them. This early
stopping can reduce the total compute needed to find a strong configuration by a
large factor. Tune reports the metrics and checkpoints from every trial, making
it straightforward to select the best model and to understand how each
hyperparameter affected the result.
