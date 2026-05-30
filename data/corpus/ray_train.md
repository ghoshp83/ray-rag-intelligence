# Ray Train

Ray Train is the distributed training library in Ray. It wraps popular deep
learning and gradient-boosting frameworks so that a training loop written for a
single machine can be scaled out to many workers with minimal changes. The
library handles the hard parts of distributed training: setting up worker
processes, sharding data, configuring the communication backend, and saving
checkpoints.

The central concept is the Trainer. A user supplies a training function that
runs on each worker, along with a scaling configuration that declares how many
workers are needed and whether each requires a GPU. Ray Train launches the
workers, runs the function in parallel, and coordinates gradient synchronisation
through the underlying framework, such as PyTorch Distributed Data Parallel.

Checkpointing is treated as a first-class concern. During training, checkpoints
are reported back to Ray and stored in a configurable location, which makes runs
resumable after a failure and enables comparison across trials. Because Ray Train
shares the same cluster and data abstractions as Ray Data and Ray Tune, a
pipeline can stream preprocessed data into training and then hand the resulting
checkpoints to a hyperparameter search without leaving the Ray runtime. This
tight integration is what lets a team move from a laptop prototype to a
multi-node training job by changing only the scaling configuration.
