# Fault Tolerance in Ray Core

Ray Core keeps a long-running job alive when individual workers fail. Every
object in the distributed store records its lineage — the task that produced it
— so if a worker is lost and an object it held is needed again, Ray can
reconstruct that object by re-executing the originating task rather than failing
the whole job. This lineage-based recovery is automatic and invisible to most
application code.

Tasks that error can be retried a configurable number of times, and actors can
be restarted up to a set limit, so a transient crash does not terminate the
computation. Because the global control store on the head node holds cluster
metadata, the head node is the critical component, and production deployments
protect it with persistence so cluster state survives a restart.

This recovery operates at the level of generic tasks and actors. It is the
substrate that higher libraries build on, but it is not itself the training,
serving, or tuning logic those libraries add on top.
