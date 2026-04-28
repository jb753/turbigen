"""Test property coverage between Block and MeanLine."""

import numpy as np
import ember.block
import ember.fluid
import turbigen.meanline_new


def test_meanline_block_property_parity():
    """Verify MeanLine properties match Block properties for numpy array outputs."""

    # 1) Initialize Block of shape (2,)
    block = ember.block.Block(shape=(2,))
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    block.set_fluid(fluid)

    # 2) Initialize MeanLine(nrow=1) - creates 2 stations
    meanline = turbigen.meanline_new.MeanLine(n_row=1)
    meanline.set_fluid(fluid)

    # 3) Set flow field using set_xrt and set_conserved
    # Set coordinates (x, r, t)
    x_values = np.array([0.0, 0.1], dtype=np.float32)
    r_values = np.array([0.5, 0.55], dtype=np.float32)
    t_values = np.array([0.0, 0.0], dtype=np.float32)

    block.set_xrt(x_values, r_values, t_values)

    # Set coordinates on each MeanLine station individually
    for i in range(2):
        meanline[i].set_x(x_values[i])
        meanline[i].set_r_rms(r_values[i])
        meanline[i].set_t(t_values[i])

    # Set conserved variables (rho, rhoVx, rhoVr, rhorVt, rhoe)
    rho = np.array([1.2, 1.15], dtype=np.float32)
    rhoVx = np.array([120.0, 115.0], dtype=np.float32)
    rhoVr = np.array([0.0, 0.0], dtype=np.float32)
    rhorVt = np.array([30.0, 31.625], dtype=np.float32)  # rho * r * Vt
    rhoe = np.array([300000.0, 295000.0], dtype=np.float32)

    block.set_conserved(rho, rhoVx, rhoVr, rhorVt, rhoe)
    meanline.set_conserved(rho, rhoVx, rhoVr, rhorVt, rhoe)

    # 4) Loop over all properties of Block
    # Discover all properties by inspecting the Block class
    block_properties = []
    for name in dir(ember.block.Block):
        # Skip private/magic methods
        if name.startswith("_"):
            continue
        # Check if it's a property
        attr = getattr(ember.block.Block, name, None)
        if isinstance(attr, property):
            block_properties.append(name)

    # Properties that are 3D-specific and not applicable to mean-lines
    skip_properties = {
        "ri",
        "rj",
        "rk",  # Face-averaged radii for 3D blocks
        "dAi",
        "dAj",
        "dAk",  # Face areas for 3D blocks
        "dAi_mag",
        "dAj_mag",
        "dAk_mag",  # Face area magnitudes for 3D blocks
        "vol",
        "dl_min",
        "ell",  # Volume and length properties for 3D blocks
        "r_cell",  # Cell-centered coordinates for 3D blocks
    }

    # Track results
    tested_properties = []
    successful_properties = []
    missing_properties = []
    shape_mismatch_properties = []
    skipped_properties = []

    # 5) Test each property
    for prop_name in sorted(block_properties):
        # Skip 3D-specific properties
        if prop_name in skip_properties:
            skipped_properties.append(prop_name)
            continue
        # Try to access property on Block
        try:
            block_value = getattr(block, prop_name)
        except Exception:
            # Skip properties that raise errors on Block access
            continue

        # Only consider properties that return numpy arrays
        if not isinstance(block_value, np.ndarray):
            continue

        tested_properties.append(prop_name)

        # Try to access same property on MeanLine
        try:
            meanline_value = getattr(meanline, prop_name)
        except (AttributeError, NotImplementedError):
            missing_properties.append(prop_name)
            continue
        except Exception as e:
            # Some other error occurred
            missing_properties.append(f"{prop_name} (error: {type(e).__name__})")
            continue

        # Verify it returns a numpy array
        if not isinstance(meanline_value, np.ndarray):
            missing_properties.append(f"{prop_name} (not array)")
            continue

        # Check shapes match
        if block_value.shape != meanline_value.shape:
            shape_mismatch_properties.append(
                f"{prop_name} (Block: {block_value.shape}, MeanLine: {meanline_value.shape})"
            )
            continue

        successful_properties.append(prop_name)

    # 6) Print concise summary
    print("\n" + "=" * 70)
    print("PROPERTY COVERAGE SUMMARY")
    print("=" * 70)
    print(f"Total properties tested: {len(tested_properties)}")
    print(f"Skipped (3D-specific): {len(skipped_properties)}")
    print(f"Successfully accessible on both: {len(successful_properties)}")
    print(f"Missing from MeanLine: {len(missing_properties)}")
    print(f"Shape mismatches: {len(shape_mismatch_properties)}")

    if missing_properties:
        print("\nMISSING PROPERTIES:")
        for prop in missing_properties:
            print(f"  - {prop}")

    if shape_mismatch_properties:
        print("\nSHAPE MISMATCHES:")
        for prop in shape_mismatch_properties:
            print(f"  - {prop}")

    if successful_properties:
        print(f"\nSUCCESSFUL PROPERTIES ({len(successful_properties)}):")
        # Print in columns for compactness
        cols = 4
        for i in range(0, len(successful_properties), cols):
            props_row = successful_properties[i : i + cols]
            print("  " + ", ".join(f"{p:20s}" for p in props_row))

    print("=" * 70)

    # Assert that ALL properties are available on MeanLine
    assert len(missing_properties) == 0, (
        f"MeanLine is missing {len(missing_properties)} properties that are available on Block. "
        f"Missing: {missing_properties}"
    )

    # Assert that there are no shape mismatches
    assert len(shape_mismatch_properties) == 0, (
        f"Found {len(shape_mismatch_properties)} properties with shape mismatches. "
        f"Mismatches: {shape_mismatch_properties}"
    )
