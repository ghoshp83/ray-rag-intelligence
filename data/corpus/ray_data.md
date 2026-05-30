# Ray Data

Ray Data is the distributed data-processing library in the Ray ecosystem. It is
designed for the last-mile data loading and transformation that feeds machine
learning workloads: reading large datasets, applying preprocessing, and running
batch inference across a cluster. Unlike a general-purpose dataframe library, Ray
Data streams blocks of data through a pipeline so that a dataset far larger than
the memory of any single machine can still be processed.

The core abstraction is the Dataset, which represents a distributed collection of
rows partitioned into blocks. Transformations such as map, filter, and
map_batches are applied lazily and executed in parallel across the available
cluster resources. The map_batches operation is the workhorse for batch
inference: it can take a stateful callable class, which loads a model once per
worker and then reuses it across many batches, amortising the expensive model
initialisation cost.

A typical batch-inference job reads input files, decodes and preprocesses each
record, runs a model over batches, and writes the predictions back out. Because
the work is partitioned, throughput scales roughly linearly with the number of
CPUs or GPUs added to the cluster. Ray Data integrates directly with Ray Train
for distributed training and with Ray Serve for online serving, so the same data
abstractions flow through the entire machine learning lifecycle without copying
data between separate systems.
