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
     n_step: 500
     n_step_avg: 100
     nstep_damp: -1

   post_process:
     convergence:
     metadata:

Log output
==========

.. code-block:: none

    TURBIGEN v2.0.0
    Starting at 2025-04-09T22:14:50
    Working directory: /home/jb753/python/turbigen/runs/test_0030
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
    499: tpnps=1.572e-07, remaining=0m0s
      block 0: 7.00e-05 2.49e-02 4.53e-04 2.82e-02 6.18e+00
    Elapsed time 8.46s
    Average tpnps=1.572e-07
    mdot_in/out=5.38/5.57, err=-3.4%
    Running post function Convergence(dn_smooth=0, rtol_loss=0.01)
    Running post function Metadata()
    Variable  Nominal  Actual  Err_abs  Err_rel/%
    ---------------------------------------------
    Alpha[0]       40    40.2    -0.15      -0.39
    Alpha[1]      -65   -63.5     -1.5        2.3
         Ma2      0.6   0.554   0.0462        7.7
          Yh    0.029  0.0753  -0.0463       -160
         htr     0.99    0.99        0  -1.65e-06
     span[0]     0.01    0.01        0          0
     span[1]    0.011   0.011        0          0
    Efficiency/%: eta_tt=-0.0, eta_ts=-0.5

Plots
=====

.. image:: profile.yaml_post_1.svg
   :width: 100%

.. image:: profile.yaml_post_2.svg
   :width: 100%


