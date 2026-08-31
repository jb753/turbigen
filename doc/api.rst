.. _api:

Python API
==========

This page documents the objects a mean-line design works with directly. The
:doc:`tutorial` walks through writing a design; here is the reference for the
classes it uses.

.. _api-meanline:

The mean-line flow field
------------------------

A :class:`~turbigen.meanline.MeanLine` is the nominal, station-averaged flow
field *and* the annulus geometry at the inlet and outlet of every blade row. It
is the object a :class:`~turbigen.design.MeanLineDesign` builds in ``forward``
and reads back in ``backward``, and the one every later stage --- annulus,
blades, mesher, post-processing --- consumes.

It *is* an :class:`ember.block.Block` of shape ``(2, n_row)``, so every flow
property ember defines is available on it directly and vectorised over both
axes: :attr:`~ember.block.Block.rho`, :attr:`~ember.block.Block.Po`,
:attr:`~ember.block.Block.Ma`, :attr:`~ember.block.Block.Ma_rel`,
:attr:`~ember.block.Block.Alpha`, and the rest. Those are documented in
`ember's Block reference <https://ember-cfd.org/dev/api/block.html>`_; only what
the mean line adds on top is listed below.

Shape and indexing
^^^^^^^^^^^^^^^^^^^

* axis 0 is the two stations of a blade row, inlet (``0``) and outlet (``1``);
* axis 1 is the blade row.

Indexing follows numpy and returns views that share storage, so a write
propagates back to the parent:

* ``ml[0]`` and ``ml[1]`` are the inlet and outlet planes of *every* row, each
  of shape ``(n_row,)``;
* ``ml[:, i]`` --- equivalently :meth:`ml.row(i) <turbigen.meanline.MeanLine.row>`
  --- is row ``i``, of shape ``(2,)``;
* ``ml[0, i]`` and ``ml[1, i]`` are that row's inlet and outlet stations.

The property :attr:`ml.flat <ember.block.Block.flat>` is a writeable
``(2 * n_row,)`` view of the stations in streamwise order, from machine inlet
to machine outlet. That is the handle for anything indexed by station rather
than by row, the annulus in particular.

:attr:`ml.inlet <turbigen.meanline.MeanLine.inlet>` and
:attr:`ml.outlet <turbigen.meanline.MeanLine.outlet>` are the first and last
stations in streamwise order --- the machine inlet and outlet. Not to be
confused with ``ml[0]`` and ``ml[1]``, which are the inlet and outlet stations
of *all* rows.

Building a mean line
^^^^^^^^^^^^^^^^^^^^^

A design gets an empty mean line of the right size from
:meth:`~turbigen.design.MeanLineDesign.allocate` and fills it in. The state is
defined through setter methods, which keep it consistent:

* a thermodynamic state, from ember: :meth:`~ember.block.Block.set_P_T`,
  :meth:`~ember.block.Block.set_P_s`, :meth:`~ember.block.Block.set_h_s`;
* an absolute-frame velocity: :meth:`~ember.block.Block.set_Vx`,
  :meth:`~ember.block.Block.set_Vr`, :meth:`~ember.block.Block.set_Vt`;
* the annulus, either as area directly with
  :meth:`~turbigen.meanline.MeanLine.set_Am`, or through one of the helpers
  :meth:`~turbigen.meanline.MeanLine.set_span_htr`,
  :meth:`~turbigen.meanline.MeanLine.set_span_r_rms`,
  :meth:`~turbigen.meanline.MeanLine.set_span_r_mid`;
* the reference-frame angular velocity,
  :meth:`~turbigen.meanline.MeanLine.set_Omega` (broadcast over all stations) or
  :meth:`~turbigen.meanline.MeanLine.set_Omega_row` (one per row). Unlike
  ember, where a block spins as a whole, ``Omega`` is stored per station here,
  so a stator row can sit at zero next to a spinning rotor.

Two traps:

* :meth:`~turbigen.meanline.MeanLine.set_Am` wants ``mdot / rho / Vm``. A design
  that writes ``mdot / rho / Vx`` is right only where the flow is axial.
* The area is the true area of the annular surface the flow crosses,
  ``2 * pi * r_mid * span``, which at a pitch angle exceeds the
  ``pi * (r_cas**2 - r_hub**2)`` that an axial view would give.

Reading derived quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the state is stored, ``backward`` and the post-processing read derived
quantities straight off the object:

* annulus geometry: :attr:`~turbigen.meanline.MeanLine.r_hub`,
  :attr:`~turbigen.meanline.MeanLine.r_cas`,
  :attr:`~turbigen.meanline.MeanLine.r_mid`,
  :attr:`~turbigen.meanline.MeanLine.span`,
  :attr:`~turbigen.meanline.MeanLine.htr`,
  :attr:`~turbigen.meanline.MeanLine.mdot`;
* overall performance: :attr:`~turbigen.meanline.MeanLine.PR_tt`,
  :attr:`~turbigen.meanline.MeanLine.PR_ts`,
  :attr:`~turbigen.meanline.MeanLine.eta_tt`,
  :attr:`~turbigen.meanline.MeanLine.eta_ts`;
* everything ember derives from the stored state --- stagnation quantities,
  Mach numbers, flow angles, and their relative-frame counterparts through
  ``Omega``.

:meth:`~turbigen.meanline.MeanLine.ref` returns the station of a row taken as
representative for wall spacings and Reynolds numbers: the end with the smaller
meridional flow area.

Precision: the storage datum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A mean line stores its state as float32, measured against its fluid's entropy
and internal-energy datum, which defaults to 1 bar and 300 K. Near ambient that
is fine. A machine running hot enough or at high enough pressure stores a large
internal energy with the kinetic energy as a small correction on top, and loses
the latter to rounding. Such a design should move the datum, before it
allocates, to the inlet conditions it already knows::

    ml = self.allocate(fluid.change_datum(P_dtm=self.Po1, T_dtm=self.To1))

Reference *scales* are a separate matter and do not help here: floating-point
precision is invariant under scaling. Scales matter to the grid a solver
iterates on, and :meth:`~turbigen.meanline.MeanLine.referenced_fluid` supplies
them there.

Serialisation
^^^^^^^^^^^^^

A mean line is a result, not a config node: it has no ``type`` and never
appears among a configuration's keys. It does round-trip, though, so a finished
run can be read back without repeating the CFD.
:meth:`~turbigen.meanline.MeanLine.to_dict` and
:meth:`~turbigen.meanline.MeanLine.from_dict` carry the eight-quantity
:attr:`~turbigen.meanline.MeanLine.STATE`. Pressure and temperature are stored
rather than the conserved variables, because conserved energy is measured from
a datum and would be silently reinterpreted if copied into a block with a
different one.

What is deliberately absent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A mean line leaves four of the keys it inherits from
:class:`ember.block.Block` unset --- ``x``, ``t``, ``mu_turb`` and ``wdist``
--- and they raise on read. A mean line has no spatial mesh, no time, and no
turbulence field; the properties exist on the base class but mean nothing here.

Reference
^^^^^^^^^

.. autoclass:: turbigen.meanline.MeanLine
   :members:
