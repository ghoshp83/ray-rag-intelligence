# Observing a Ray Application

Ray ships a dashboard that shows the live state of a cluster: the nodes and
their resource usage, the actors and tasks currently running, and the logs each
worker emits. It is the first place to look when a job is slower than expected or
a deployment is not behaving, because it makes the otherwise invisible
distributed execution concrete.

For performance work, Ray can record a timeline of task and actor events that is
viewable as a trace, revealing where time is spent and whether work is actually
running in parallel or serialising on a bottleneck. Metrics such as resource
utilisation and task counts are exported in a standard format so they can be
scraped into an external monitoring system and alerted on.

Structured application logging complements these built-in views: emitting one
record per meaningful event, with consistent fields, lets operators query what
happened across a run rather than reading raw console output. Observability is
about explaining behaviour after the fact, not about scheduling or executing the
work itself.
