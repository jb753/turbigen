# Geometry generation method

#### Flow angles

Applying the Euler work equation across the rotor,
$$
h_{03} - h_{02} = (U V_\theta)_3 - (U V_\theta)_2 \, .
$$
For a constant mean radius, we set $U_3=U_2=U$, then putting $V_\theta = V_x
\tan \alpha$ yields,
$$
h_{03}-h_{02} = U V_{x2} \left(\frac{V_{x3}}{V_{x2}} \tan\alpha_3 - \tan\alpha_2 \right) \, ,
$$
and dividing by $U^2$ to put in dimensionless terms,
$$
\psi = \phi \left(\zeta_3 \tan\alpha_3 - \tan\alpha_2 \right) \, . \tag{I}
$$
Suppose the inlet swirl, $\alpha_1$, is given. If we choose a value for stage exit swirl,
$\alpha_3$, then Eqn.\ (I) allows us to solve for the vane exit swirl
$\alpha_2$ and we have absolute flow angles throughout the stage.

#### Velocities

The next step is to calculate velocities throughout the stage (all
non-dimensionalised by the blade speed $U$). With prescribed axial velocity
ratios,
$$
\frac{V_{x1}}{U} = \zeta_1 \phi\,, \quad \frac{V_{x2}}{U} = \phi\,, \quad \frac{V_{x3}}{U} = \zeta_3 \phi \, .
$$
Tangential velocities in stationary and rotor-relative frames are given by,
$$
\frac{V_\theta}{U} = \frac{V_x}{U} \tan \alpha \,, \quad \frac{V_\theta^\mathrm{rel}}{U} =  \frac{V_\theta}{U} -1 \,,
$$
then velocity magnitudes and relative flow angles are trivially calculable.

#### Compresibility
