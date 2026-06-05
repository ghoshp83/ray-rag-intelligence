# Ray Core

Ray Core is the foundation the rest of the Ray ecosystem is built on. It turns a
single Python program into a distributed one through two primitives: tasks and
actors. A task is a stateless function annotated so that Ray can schedule it on
any worker in the cluster and run many copies in parallel. An actor is a stateful
class whose instance lives on one worker and processes calls one at a time, which
is how Ray models things like a loaded model or a running counter.

Both primitives return object references rather than blocking on a result. These
references point into the distributed object store, a shared-memory layer that
holds large objects once per node and lets multiple workers read them without
copying. Passing a reference instead of the value is what keeps data movement
cheap when a big array is reused across many tasks.

The same code runs unchanged whether Ray is started on a laptop with a few cores
or on a cluster with hundreds of nodes; the scheduler places work wherever
resources are free. Ray Data, Ray Train, Ray Tune, and Ray Serve are all
libraries layered on these Core primitives, which is why they compose cleanly
inside one application.
