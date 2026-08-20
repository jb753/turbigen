===============
Turbine cascade
===============

A single stationary blade row, taking flow in at 40 degrees and turning it to
65 degrees the other way, leaving at a Mach number of 0.6. With a hub-to-tip
ratio of 0.95 the flow is nearly two-dimensional, which makes this the cheapest
case that still exercises every stage of the program: mean line, annulus,
blade, mesh, solve and report.

Two things in the configuration are worth reading for their own sake. The
annulus is specified by *aspect ratio* rather than by a chord in metres, which
is the number a designer carries between machines; the chord follows from the
span the mean line chose. And the viscosity is set by the ``Re_surf`` iterator
rather than by hand, because a Reynolds number cannot be inverted for a
viscosity without a design, and a design needs a viscosity to exist first. That
circularity is why it is an iterator --- but one that closes on the design
alone, so it costs no CFD and every verb honours it.

.. turbigen-example:: turbine_cascade
