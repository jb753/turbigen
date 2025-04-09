===============
Turbine cascade
===============

Input file
==========

.. code-block:: yaml

   # Turbine cascade
   #
   # I like beans.

   workdir: runs/test_*

   inlet:
     Po: 1e5
     To: 300.
     cp: 1005.
     gamma: 1.4

   Re_surf: 4e5

   mean_line:
     type: turbine_cascade
     span: [0.01, 0.011]
     Alpha: [40., -65.0]
     Ma2: 0.6
     Yh: 0.029
     htr: 0.99

   annulus:
     AR_gap: [1.0, 1.0]
     AR_chord: 2.0

   mesh:
     type: h
     dm_LE: 0.001
     ni_TE: 9
     dm_TE: 0.03
     yplus: 30.0
     resolution_factor: 0.5

   blades:
       - spf: 0.5
         q_thick: [0.05, 0.12, 0.3, 0., 0.00, 0.18]
         q_camber: [10., -2., 1.0, 1.0, 0.0]

   nblade:
     - Co: 0.7

   solver:
     type: emb
     n_step: 100
     n_step_avg: 100
     nstep_damp: -1

Log output
==========

.. code-block:: none

    TURBIGEN v2.0.0
    Starting at 2025-04-09T15:33:49
    Working directory: /home/jb753/python/turbigen-dev/runs/test_0084
    Inlet: PerfectState(P=1.000 bar, T=300.0 K)
    MeanLine(
        Po=[1.     0.9927] bar,
        To=[300. 300.] K,
        Ma=[0.311 0.6  ],
        Vx=[81.9 85.1],
        Vr=[0. 0.],
        Vt=[  68.7 -182.4],
        Vt_rel=[  68.7 -182.4],
        Al=[ 40. -65.],
        Al_rel=[ 40. -65.],
        rpm=[0. 0.],
        mdot=[5.66 5.66] kg/s
        )
    Designing annulus...
    FixedAR(nrow=1, x=[0.002625], r=[0.995], AR=[2.])
    Designing blades...
    Nblade: [473]
    Tip gaps: [0.]
    Re_surf=[4e+05]
    Generating mesh...
    Making an H-mesh...
    ncell/1e6=0.1
    Applying 2D guess...
    Initialising native solver...
    Patitioning onto 1 processors...
    Starting the main time-stepping loop...
    99: tpnps=6.696e-08, remaining=0m0s
      block 0: 1.21e-05 4.43e-03 1.29e-04 7.11e-03 1.19e+00
    Elapsed time 3.60s
    Average tpnps=3.349e-07
    mdot_in,out =5.66,5.66, err=-0.0%
    Variable  Nominal  Actual    Err_abs  Err_rel/%
    -----------------------------------------------
    Alpha[0]       40    40.3      -0.31      -0.78
    Alpha[1]      -65   -64.9     -0.099       0.15
         Ma2      0.6     0.6   0.000126     0.0211
          Yh    0.029  0.0298  -0.000811       -2.8
         htr     0.99    0.99          0          0
     span[0]     0.01    0.01          0          0
     span[1]    0.011   0.011          0          0
    Efficiency/%: eta_tt=0.0, eta_ts=0.0
