from turbigen.tables import make_tables

fluid_name = "water"  # Fluid name in CoolProp
smin = 7308.0  # Minimum entropy in J/kg/K
smax = 7600.0  # Maximum entropy in J/kg/K
Pmin = 37746.0  # Minimum pressure in Pa
Tmax = 550.0  # Maximum temperature in K
ni = 200  # Number of interpolation points in each direction
new_npz_path = "water_new.npz"  # Path to save the new tables

make_tables(fluid_name, smin, smax, Pmin, Tmax, ni, new_npz_path)
