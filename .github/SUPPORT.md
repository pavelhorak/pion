# Getting help

**A question, an idea, or "should this work?"** →
[Discussions](https://github.com/pavelhorak/pion/discussions). No question about
getting Pion running is too small; if the README made you guess, that is a
documentation bug and worth telling us about.

**A wrong reply, a desync, data loss, a crash** →
[open an issue](https://github.com/pavelhorak/pion/issues/new/choose). These are
the highest priority, in that order.

**A surprising benchmark number** → the performance issue template asks several
questions about machine state before the numbers. They are not busywork: most
surprising numbers turn out to be a leaked process or a cold cache, and the
template is the fastest route to finding out which.

**Security** → see [SECURITY.md](../SECURITY.md). Do not open a public issue for
a vulnerability.

## What to expect

Pion is maintained by one person alongside a day job. Crash, desync and
data-loss reports get looked at first and fastest. Feature requests are read and
may sit for a while; a request that describes a workload rather than an API is
more likely to move.

There is no SLA, and it would be dishonest to imply one. Pion is
Apache-2.0 and the gates are runnable locally, so the project is inspectable
and forkable regardless of how responsive the maintainer
is in any given month.
