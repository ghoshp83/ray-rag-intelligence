# Ray Serve

Ray Serve is the model-serving library in Ray, built for composing multiple
models into a single online application. It is framework-agnostic and is designed
to scale each part of an inference pipeline independently. This makes it a strong
fit for systems that combine several models, such as a retrieval-augmented
generation service that runs a retriever, a reranker, and a language model in
sequence.

The core abstraction is the deployment, a class or function that Serve manages as
a group of replicas. Each deployment declares its own resource requirements and
replica count, so a lightweight preprocessing step and a heavy GPU model can be
scaled separately rather than forcing the whole pipeline to the same footprint.
Deployments are bound together into a deployment graph, where the output of one
deployment becomes the input to the next.

Serve exposes the graph behind an HTTP endpoint, commonly using a FastAPI ingress
to handle request parsing and validation. It supports request batching, which
groups incoming requests so that a model can process them together for higher
throughput, and autoscaling, which adjusts the number of replicas based on load.
Because Serve runs on the same Ray cluster as Ray Data, Ray Train, and Ray Tune,
a model can move from training to production without changing infrastructure,
which is central to Ray's promise of one runtime from laptop to cluster.
