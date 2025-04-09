# Geometry generation method

#### Flow angles

Applying the Euler work equation across the rotor,
$$
h_{03} - h_{02} = (U V_\theta)_3 - (U V_\theta)_2 \, .
$$
For a constant mean radius, we set $U_3=U_2=U$, then putting $V_\theta = V_x
\tan \alpha$ yields,
$$
h_{03}-h_{02} = U V_{x2} \left(\frac{V_{x3}}{V_{x2}} \tan\alpha_3
- \tan\alpha_2 \right) \, ,
$$
and dividing by $U^2$ to put in dimensionless terms,
$$
\psi = \phi \left(\zeta_3 \tan\alpha_3 - \tan\alpha_2 \right) \, . \tag{I}
$$
Suppose the inlet swirl, $\alpha_1$, is given. If we choose a value for stage
exit swirl, $\alpha_3$, then Eqn.\ (I) allows us to solve for the vane exit
swirl $\alpha_2$ and we have absolute flow angles throughout the stage.

#### Velocities

The next step is to calculate velocities throughout the stage (all
non-dimensionalised by the blade speed $U$). With prescribed axial velocity
ratios,
$$
\frac{V_{x1}}{U} = \zeta_1 \phi\,, \quad \frac{V_{x2}}{U}
= \phi\,, \quad \frac{V_{x3}}{U} = \zeta_3 \phi \, .
$$
Tangential velocities in stationary and rotor-relative frames are given by,
$$
\frac{V_\theta}{U} = \frac{V_x}{U} \tan \alpha \,,
\quad \frac{V_\theta^\mathrm{rel}}{U} =  \frac{V_\theta}{U} -1 \,,
$$
then velocity magnitudes and relative flow angles are trivially calculable.

#### Compressibility

Up to this point, we have only considered the kinematics of the turbine stage.
Now we bring in fluid properties and compressibility, For a perfect gas, there
exists a compressible flow relation,
$$
\newcommand{\Ma}{\mathit{M\kern-.05ema}}
\frac{V}{\sqrt{c_p T_0}} = \sqrt{\gamma -1}\, \Ma
\left(1 + \frac{\gamma - 1}{2} \Ma^2 \right)^{-\tfrac{1}{2}} \, , \tag{II}
$$
where $\gamma$ is the ratio of specific heats and $c_p$ the specific heat at
constant pressure of the fluid.
As the stator does no work, the vane exit
stagnation temperature is equal to the inlet stagnation temperature. We can
then write,
$$
\frac{U}{\sqrt{c_p T_{01}}} = \frac{U}{V_2}\frac{V_2}{\sqrt{c_p T_{02}}} =
\frac{1}{V_2/U}\sqrt{\gamma -1}\, \Ma_2\left(1 + \frac{\gamma - 1}{2} \Ma_2^2 \right)^{-\tfrac{1}{2}} \, . \tag{III}
$$
Given a non-dimensional velocity magnitude calculated in the previous step,
$V_2/U$, Eqn. (III) evaluates the non-dimensional blade speed $U/\sqrt{c_p
T_{01}}$ for a prescribed vane exit Mach number $\Ma_2$.

The stagnation temperature downstream of the rotor depends on the stage loading
coefficient,
$$
\frac{c_p T_{03}}{U^2} = \frac{c_p T_{01}}{U^2} - \psi \, .
$$
Combining the values of ${c_p T_{0}}/{U^2}$ with values of $V/U$ for each
station, and inverting Eqn. (II) yields Mach numbers throughout the stage.
Taking ratios of each ${c_p T_{0}}/{U^2}$ with the inlet value yields
stagnation temperature ratios for each station independent of blade speed.

#### Loss

The annulus line depends on the level of blockage and hence loss of stagnation
pressure or entropy creation within the turbine stage. We choose a polytropic
efficiency, $\eta$, to characterise loss as it directly relates to the gas
turbine cycle performance. Taking the logarithm of the definition of turbine polytropic efficiency,
$$
\frac{T_{03}}{T_{01}} = \left(\frac{p_{03}}{p_{01}}\right)
^{\eta\tfrac{\gamma - 1}{\gamma}} \quad \Rightarrow \quad
\log\frac{T_{03}}{T_{01}} =
\eta\frac{\gamma - 1}{\gamma}\log\frac{p_{03}}{p_{01}}\ ,
$$
with the usual perfect gas formula for entropy change,
$$
\Delta s = c_p \log\frac{T_{03}}{T_{01}} - R \log\frac{p_{03}}{p_{01}}\ , \tag{IV}
$$
combining yields the following expression for entropy creation across a turbine
stage,
$$
\frac{\Delta s}{c_p} =
\left(1-\frac{1}{\eta}\right)\log\frac{T_{03}}{T_{01}} \ .
$$

How much of the calculated loss occurs in the stator or rotor is a free
variable; we arbitrarily choose an equal split. Substituting back values of
entropy, $\Delta s/c_p$ and stagnation temperature into Eqn. (IV) yields
stagnation pressures at each station. The mean-line flow quantities are now
fully-defined.

#### Annulus line

We now need to set the annulus line, which in a mean-line sense corresponds to
axial velocity density ratios at each station, or equivalently annulus area
ratios. The compressible flow relation for non-dimensional mass flow is,
$$
\frac{\dot{m}\sqrt{c_p T_0}}{A p_0} =
\frac{\gamma}{\sqrt{\gamma -1}}\, \Ma
\left(1 + \frac{\gamma - 1}{2} \Ma^2 \right)
^{-\tfrac{1}{2}\tfrac{\gamma + 1}{\gamma - 1}} = Q(\Ma,\gamma)\ .
$$
So far, we know $c_p$, $T_0$ and $p_0$ from the left-hand side, and can
evaluate the right-hand side $Q$ with known $\Ma$ and $\gamma$. At any given
station, we can write,
$$
A_x = \frac{\dot{m} \sqrt{c_p T_0}}{p_0\,Q(\Ma,\gamma)\cos \alpha}\ ,
$$
then forming ratios and using conservation of mass gives,
$$
\frac{A_x}{A_{x1}} = \sqrt{\frac{T_{0}}{T_{01}}} \frac{p_{01}}{p_0}
\frac{Q(\Ma_1,\gamma)}{Q(\Ma,\gamma)} \frac{\cos \alpha_1}{\cos \alpha}\ .
$$
For constant mean radius, annulus area ratios are identical to blade height
ratios so that $\Delta r/\Delta r_1 = A_x/A_{x1}$.

#### Summary

We began with a set of *aerodynamic* parameters fully defining the mean-line
flow of a turbine stage at constant mean radius:

* $\phi$, $\psi$, $\alpha_1$, $\alpha_3$, $\zeta$ describing the velocity triangles;
* $\Ma_2$ and $\gamma$ to describe compressibility effects;
* $\eta$ to specify irreversibility.

Using the Euler work equation and assuming a perfect gas, we have derived a set of analytic equations that convert the
*aerodynamic* parameters into a set of *geometric* parameters
sufficient to construct a non-dimensional cascade shape:

* $\alpha_2$, the inter-stage swirl;
* $U/\sqrt{c_p T_01}$, a blade speed;
* $\Delta r/\Delta r_1$ at each station.

The following section discusses geometry generation at a specified degree of
reaction, $\Lambda$, which must be done numerically.

## Specifying reaction

In a turbomachinery design context, it is more convenient to specify degree of
reaction instead of the exit flow angle of a turbine stage. Reaction directly
controls the balance of loading between stator and rotor; $\Lambda\approx0.5$
is a common design philosophy to equalise peak velocities in stator and rotor,
minimising loss.
