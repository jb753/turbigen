"""Compare TS4 gas tables with known good."""

import numpy as np
import os
from turbigen.tables import make_tables
from tempfile import mkdtemp
import pytest

import matplotlib.pyplot as plt

# Look for test data in a directory at same level as this script
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Parameters for the test table
FLUID_NAME = "water"
SMIN = 7308.0
SMAX = 7600.0
PMIN = 37746.0
TMAX = 550.0
NI = 200


def test_compare_tables():
    # Load good tables

    tables_good_npz = os.path.join(DATA_DIR, "water_table_known_good.npz")
    if not os.path.exists(tables_good_npz):
        pytest.skip(f"Known good table {tables_good_npz} not found")
    tables_good = np.load(tables_good_npz)
    # tables_good = np.load(os.path.join(DATA_DIR, "water_coarse.npz"))

    # Get temporary director to save new tables
    tmp_dir = mkdtemp()
    new_npz_path = os.path.join(tmp_dir, "water_new.npz")

    # Save and load in new tables
    make_tables(FLUID_NAME, SMIN, SMAX, PMIN, TMAX, NI, new_npz_path)
    tables_new = np.load(new_npz_path)

    # loop over vars in the tables
    for var_name in tables_good:
        table_good = tables_good[var_name]
        table_new = tables_new[var_name]

        # Check shape
        if not (table_good.shape == table_new.shape):
            print(f"Shape mismatch: {table_good.shape}, {table_new.shape}")
            assert False

        # Check values
        err_rel = np.abs(table_new / table_good - 1.0)
        if np.mean(err_rel) > 1e-4:
            print(
                f"  {var_name}, good_av={np.nanmean(table_good)},"
                f" new_av={np.nanmean(table_new)}, err_max={err_rel.max()},"
                f" err_av={np.nanmean(err_rel)}"
            )
            assert False


def test_compare_hydrogen():
    # Load good tables
    tables_good_npz = os.path.join(DATA_DIR, "hydrogen_table_known_good.npz")
    if not os.path.exists(tables_good_npz):
        pytest.skip(f"Known good table {tables_good_npz} not found")
    tables_good = np.load(tables_good_npz)
    # tables_good = np.load(os.path.join(DATA_DIR, "water_coarse.npz"))

    # Get temporary director to save new tables
    tmp_dir = mkdtemp()
    new_npz_path = os.path.join(tmp_dir, "hydrogen_new.npz")

    # Save and load in new tables
    smin = 3200.0
    smax = 4100.0
    Pmin = 1.3e6
    Tmax = 35.8
    fluid_name = "Hydrogen"
    make_tables(fluid_name, smin, smax, Pmin, Tmax, 10, new_npz_path)
    tables_new = np.load(new_npz_path)

    # loop over vars in the tables
    for var_name in tables_good:
        table_good = tables_good[var_name]
        table_new = tables_new[var_name]

        # Check shape
        if not (table_good.shape == table_new.shape):
            print(f"Shape mismatch: {table_good.shape}, {table_new.shape}")
            assert False

        # Check values
        err_rel = np.abs(table_new / table_good - 1.0)
        if np.mean(err_rel) > 1e-4:
            print(
                f"  {var_name}, good_av={np.nanmean(table_good)},"
                f" new_av={np.nanmean(table_new)}, err_max={err_rel.max()},"
                f" err_av={np.nanmean(err_rel)}"
            )
            assert False


def plot_error():
    # Load good tables
    tables_good = np.load(os.path.join(DATA_DIR, "water_table_known_good.npz"))
    # tables_good = np.load(os.path.join(DATA_DIR, "water_coarse.npz"))

    # Get temporary director to save new tables
    tmp_dir = mkdtemp()
    new_npz_path = os.path.join(tmp_dir, "water_new.npz")

    # Save and load in new tables
    make_tables(FLUID_NAME, SMIN, SMAX, PMIN, TMAX, NI, new_npz_path)
    tables_new = np.load(new_npz_path)

    tables_good = [tables_good[k] for k in tables_good]
    tables_new = [tables_new[k] for k in tables_new]

    # Loop over tables and plot if error large
    for i in range(23):
        xn = tables_new[3 * i]
        yn = tables_new[3 * i + 1]
        zn = tables_new[3 * i + 2]
        zg = tables_good[3 * i + 2]
        err_rel = np.abs(zn / zg - 1.0)
        if np.sum(err_rel > 0.001) or True:
            fig, ax = plt.subplots(1, 3)
            plt.grid(False)
            xg, yg = np.meshgrid(xn, yn)
            for a in ax:
                a.grid(False)
            ax[0].pcolor(xn, yn, np.gradient(np.gradient(zg, axis=0), axis=0))
            ax[1].pcolor(xn, yn, np.gradient(np.gradient(zn, axis=0), axis=0))
            ax[2].pcolor(xn, yn, err_rel)
            ax[0].set_title("Official")
            ax[1].set_title("New")
            ax[2].set_title("Error")
            plt.tight_layout()
            plt.savefig(f"table_{i}.pdf")
            plt.close()


if __name__ == "__main__":
    plot_error()
    # test_compare_tables()
