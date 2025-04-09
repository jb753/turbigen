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
    Starting at 2025-04-09T21:48:13
    Working directory: /home/jb753/python/turbigen/runs/test_0015
    Saving source code backup to /home/jb753/python/turbigen/runs/test_0015/src.tar.gz
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
    Checking mean-line conservation...
    Checking mean-line inversion...
    Designing annulus...
    <turbigen.annulus.Smooth object at 0x7f4f2f74d940>
    Designing blades...
    Nblade: [473]
    Tip gaps: [0.]
    Re_surf=[4e+05]
    Generating mesh...
    Making mesh...
    Generating an H-mesh...
    ncell/1e6=0.1
    Applying 2D guess...
    Entering embsolve run, memory usage on rank 0: 160MB
    Initialising native solver...
    Patitioning onto 1 processors...
    Elapsed time 0.26s
    Sending data to processors...
    Elapsed time 0.00s
    Starting the main time-stepping loop...
    Memory usage on rank 0: 232MB
    After allocation Memory usage on rank 0: 236MB
    100: tpnps=1.606e-07, remaining=0m6s
      block 0: 6.13e-05 2.27e-02 6.46e-04 3.60e-02 5.98e+00
    200: tpnps=1.557e-07, remaining=0m5s
      block 0: 8.17e-05 3.40e-02 4.03e-04 2.68e-02 5.71e+00
    300: tpnps=1.583e-07, remaining=0m3s
      block 0: 7.17e-05 2.00e-02 4.68e-04 2.39e-02 6.27e+00
    400: tpnps=1.629e-07, remaining=0m1s
      block 0: 7.27e-05 2.47e-02 4.51e-04 3.00e-02 6.79e+00
    499: tpnps=1.580e-07, remaining=0m0s
      block 0: 6.29e-05 2.36e-02 2.98e-04 2.41e-02 6.17e+00
    Elapsed time 8.56s
    Average tpnps=1.591e-07
    Recieving data from processors...
    Elapsed time 0.00s
    mdot_in=5.3840182404857995, mdot_out=5.573253684994447
    Mass flow error: -3.4%
    Post directory /home/jb753/python/turbigen/runs/test_0015/post True
    Running post function convergence
    Error encountered, quitting...
    Traceback (most recent call last):
      File "/home/jb753/python/turbigen/.venv/bin/turbigen", line 10, in <module>
        sys.exit(main())
                 ~~~~^^
      File "/home/jb753/python/turbigen/.venv/lib/python3.13/site-packages/turbigen/main.py", line 212, in main
        conf.design_and_run()
        ~~~~~~~~~~~~~~~~~~~^^
      File "/home/jb753/python/turbigen/.venv/lib/python3.13/site-packages/turbigen/config2.py", line 634, in design_and_run
        self.post_process_all()
        ~~~~~~~~~~~~~~~~~~~~~^^
      File "/home/jb753/python/turbigen/.venv/lib/python3.13/site-packages/turbigen/config2.py", line 660, in post_process_all
        post_func = util.load_post(post_name).post
                    ~~~~~~~~~~~~~~^^^^^^^^^^^
      File "/home/jb753/python/turbigen/.venv/lib/python3.13/site-packages/turbigen/util.py", line 629, in load_post
        mod = importlib.import_module(f".{post_type}", package="turbigen.post")
      File "/usr/lib/python3.13/importlib/__init__.py", line 88, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
               ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
      File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
      File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
    ModuleNotFoundError: No module named 'turbigen.post.convergence'

