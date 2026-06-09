# Ray Cluster Architecture

A Ray cluster is a head node plus a set of worker nodes that together form a
single logical pool of CPUs, GPUs, and memory. The head node runs the global
control store and the scheduler that places tasks and actors; worker nodes
contribute resources and execute the work. Application code is written once and
the scheduler decides which node each task runs on, so the same program runs on
a laptop and on a hundred-node cluster without changes.

The autoscaler watches pending demand and adds or removes worker nodes to match
it, scaling the cluster up under load and back down when idle to control cost.
Each node advertises its resources, and scheduling requests can ask for specific
amounts — a fractional CPU, a whole GPU, or a custom resource — so heterogeneous
hardware is used deliberately rather than uniformly.

Clusters can run on a single machine for development, on Kubernetes through
KubeRay, or on a managed platform, and the resource model is identical in every
case.
