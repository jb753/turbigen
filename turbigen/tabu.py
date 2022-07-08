"""Functions for multiobjective tabu search."""
import json
import numpy as np

np.set_printoptions(precision=3, linewidth=np.inf)


def hj_move(x, dx):
    """Generate a set of Hooke and Jeeves moves about a point.

    For a design vector with M variables, return a 2MxM matrix, each variable
    having being perturbed by elementwise +/- dx."""
    d = np.diag(dx.flat)
    return x + np.concatenate((d, -d))


def objective(x):
    yy = np.column_stack((x[:, 0], (1.0 + x[:, 1]) / x[:, 0]))
    yy[~constrain(x)] = np.nan
    return yy


def constrain(x):
    return np.all(
        (
            x[:, 1] + 9.0 * x[:, 0] >= 6.0,
            -x[:, 1] + 9.0 * x[:, 0] >= 1.0,
            x[:, 0] >= 0.0,
            x[:, 0] <= 1.0,
            x[:, 1] >= 0.0,
            x[:, 1] <= 5.0,
        ),
        axis=0,
    )


def find_rows(A, B, atol=None):
    """Get matching rows in matrices A and B.

    Return:
        logical same shape as A, True where A is in B
        indices same shape as A, of the first found row in B for each A row."""

    # Arrange the A points along a new dimension
    A1 = np.expand_dims(A, 1)

    # NA by NB mem logical where all elements match
    if atol is None:
        b = (A1 == B).all(axis=-1)
    else:
        b = np.isclose(A1, B, atol=atol).all(axis=-1)

    # A index is True where it matches any of the B points
    ind_A = b.any(axis=1)

    # Use argmax to find first True along each row
    loc_B = np.argmax(b, axis=1)

    # Where there are no matches, override to a sentinel value -1
    loc_B[~ind_A] = -1

    return ind_A, loc_B


class Memory:
    def __init__(self, nx, ny, max_points, tol=None):
        """Store a set of design vectors and their objective functions."""

        # Record inputs
        self.nx = nx
        self.ny = ny
        self.max_points = max_points
        self.tol = None

        # Initialise points counter
        self.npts = 0

        # Preallocate matrices for design vectors and objectives
        # Private because the user should not have to deal with empty slots
        self._X = np.empty((max_points, nx))
        self._Y = np.empty((max_points, ny))

    # Public read-only properties for X and Y
    @property
    def X(self):
        """The current set of design vectors."""
        return self._X[: self.npts, :]

    @property
    def Y(self):
        """The current set of objective functions."""
        return self._Y[: self.npts, :]

    def contains(self, Xtest):
        """Boolean index for each row in Xtest, True if x already in memory."""
        if self.npts:
            return find_rows(Xtest, self.X, self.tol)[0]
        else:
            return np.zeros((Xtest.shape[0],), dtype=bool)

    def get(self, ind):
        """Get the entry for a specific index."""
        return self._X[ind, :].reshape(1, -1), self._Y[ind, :].reshape(1, -1)

    def add(self, xa, ya=None):
        """Add a point to the memory."""
        xa = np.atleast_2d(xa)
        if ya is None:
            ya = np.empty((xa.shape[0], self.ny))
        else:
            ya = np.atleast_2d(ya)

        # Only add new points
        i_new = ~self.contains(xa)
        n_new = np.sum(i_new)
        xa = xa[i_new]
        ya = ya[i_new]

        # Roll downwards and overwrite
        self._X = np.roll(self._X, n_new, axis=0)
        self._X[:n_new, :] = xa
        self._Y = np.roll(self._Y, n_new, axis=0)
        self._Y[:n_new, :] = ya

        # Update points counter
        self.npts = np.min((self.max_points, self.npts + n_new))

    def lookup(self, Xtest):
        """Return objective function for design vector already in mem."""

        # Check that the requested points really are available
        if np.any(~self.contains(Xtest)):
            raise ValueError(
                "The requested points have not been previously evaluated"
            )

        return self.Y[find_rows(Xtest, self.X)[1]]

    def delete(self, ind_del):
        """Remove points at given indexes."""

        # Set up boolean mask for points to keep
        b = np.ones((self.npts,), dtype=bool)
        b[ind_del] = False
        n_keep = np.sum(b)

        # Reindex everything so that spaces appear at the end of memory
        self._X[:n_keep] = self.X[b]
        self._Y[:n_keep] = self.Y[b]

        # Update number of points
        self.npts = n_keep

    def update_front(self, X, Y):
        """Add or remove points to maintain a Pareto front."""
        Yopt = self.Y

        # Arrange the test points along a new dimension
        Y1 = np.expand_dims(Y, 1)

        # False where an old point is dominated by a new point
        b_old = ~(Y1 < Yopt).all(axis=-1).any(axis=0)

        # False where a new point is dominated by an old point
        b_new = ~(Y1 >= Yopt).all(axis=-1).any(axis=1)

        # False where a new point is dominated by a new point
        b_self = ~(Y1 > Y).all(axis=-1).any(axis=1)

        # We only want new points that are non-dominated
        b_new_self = np.logical_and(b_new, b_self)

        # Delete old points that are now dominated by new points
        self.delete(~b_old)

        # Add new points
        self.add(X[b_new_self], Y[b_new_self])

        # Return true if we added at least one point
        return np.sum(b_new_self) > 0

    def update_best(self, X, Y):
        """Add or remove points to keep the best N in memory."""

        X, Y = np.atleast_2d(X), np.atleast_2d(Y)

        in_already = self.contains(X)

        # Join memory and test points into one matrix
        Yall = np.concatenate((self.Y, Y), axis=0)
        Xall = np.concatenate((self.X, X), axis=0)

        # Sort by objective, truncate to maximum number of points
        # isort = np.argsort(Yall[:, 0], axis=0)[: self.max_points]
        _, isort = np.unique(Yall[:, 0], axis=0, return_index=True)
        npts = min(len(isort), self.max_points)
        isort = isort[:npts]
        Xall, Yall = Xall[isort], Yall[isort]

        # Reassign to the memory
        self.npts = npts
        self._X[:npts] = Xall
        self._Y[:npts] = Yall

        # If any of the input points have been added return True
        in_now = self.contains(X)
        return np.any(np.logical_and(in_now, ~in_already))

    def generate_sparse(self, nregion):
        """Return a random design vector in a underexplored region."""

        # Loop over each variable
        xnew = np.empty((1, self.nx))
        for i in range(self.nx):

            # Bin the design variable
            hX, bX = np.histogram(self.X[:, i], nregion)

            # Random value in least-visited bin
            bin_min = hX.argmin()
            bnds = bX[bin_min : bin_min + 2]
            xnew[0, i] = np.random.uniform(*bnds)

        return xnew

    def sample_random(self):
        """Choose a random design point from the memory."""
        return self.get(np.random.choice(self.npts, 1))

    def sample_sparse(self, nregion):
        """Choose a design point from sparse region of the memory."""

        # Randomly pick a component of x to bin
        dirn = np.random.choice(self.nx)

        # Arbitrarily bin on first design variable
        hX, bX = np.histogram(self.X[:, dirn], nregion)

        # Override count in empty bins so we do not pick them
        hX[hX == 0] = hX.max() + 1

        # Choose sparsest bin, breaking ties randomly
        i_bin = np.random.choice(np.flatnonzero(hX == hX.min()))

        # Logical indexes for chosen bin
        log_bin = np.logical_and(
            self.X[:, dirn] >= bX[i_bin], self.X[:, dirn] <= bX[i_bin + 1]
        )
        # Choose randomly from sparsest bin
        i_select = np.atleast_1d(np.random.choice(np.flatnonzero(log_bin)))

        return self._X[i_select], self._Y[i_select]

    def clear(self):
        """Erase all points in memory."""
        self.npts = 0

    def to_dict(self):
        """Serialise as a dictionary."""

        return {
            "nx": self.nx,
            "ny": self.ny,
            "max_points": self.max_points,
            "npts": self.npts,
            "Xflat": self._X.flatten().tolist(),
            "Yflat": self._Y.flatten().tolist(),
            "Xshape": self._X.shape,
            "Yshape": self._Y.shape,
        }

    def from_dict(self, d):
        """Deserialise from a dictionary."""

        assert d["nx"] == self.nx
        assert d["ny"] == self.ny
        assert d["max_points"] <= self.max_points

        self.npts = d["npts"]
        self._X = np.reshape(d["Xflat"], d["Xshape"])
        self._Y = np.reshape(d["Yflat"], d["Yshape"])

    def to_file(self, fname):
        with open(fname, "w") as f:
            json.dump(self.to_dict(), f)


class TabuSearch:

    MEM_KEYS = ["short", "med", "long", "ban"]

    def __init__(self, objective, constraint, nx, ny, tol, j_obj=None):
        """Maximise an objective function using Tabu search."""

        # Store objective and constraint functions
        self.objective = objective
        self.constraint = constraint

        # Store tolerance on x
        self.tol = tol

        # Default memory sizes
        self.n_short = 20
        self.n_long = 20000
        self.nx = nx
        self.ny = ny

        if j_obj is None:
            self.j_objective = range(ny)
        else:
            self.j_objective = j_obj

        self.n_med = 2000 if len(self.j_objective) > 1 else 10

        self.mem_file = None

        self.i_search = 0
        self.x0 = None
        self.y0 = None
        self.dx = None

        # Misc algorithm parameters
        self.x_regions = 3
        self.max_fevals = 500
        self.fac_restart = 0.5
        self.fac_pattern = 2.0
        self.max_parallel = 4

        self.verbose = True

        # Default iteration counters
        self.i_diversify = self.max_fevals
        self.i_intensify = [
            5,
            10,
            15,
        ]
        self.i_restart = 20
        self.i_pattern = None

        # Initialise counters
        self.fevals = 0

        # Initialise memories
        self.mem_short = Memory(nx, ny, self.n_short, self.tol)
        self.mem_med = Memory(nx, ny, self.n_med, self.tol)
        self.mem_long = Memory(nx, ny, self.n_long, self.tol)
        self.mem_ban = Memory(nx, ny, self.n_long, self.tol)
        self.mem_all = (
            self.mem_short,
            self.mem_med,
            self.mem_long,
            self.mem_ban,
        )

    def clear_memories(self):
        """Erase all memories"""
        for mem in self.mem_all:
            mem.clear()

    def initial_guess(self, x0):
        """Set current point, evaluate objective."""
        if self.verbose:
            print("INITIAL\n  x = %s" % np.array_str(x0))
        if not self.constraint(x0):
            raise Exception("Constraints not satisfied at initial point.")
        if self.mem_long.contains(x0):
            if self.verbose:
                print("  Using initial guess from memory.")
            y0 = self.mem_long.lookup(x0)
        else:
            if self.verbose:
                print("  Evaluating initial guess.")
            y0 = self.objective(x0)
            self.fevals += 1

        # Add to tabu and long memories
        self.mem_short.add(x0, y0)
        self.mem_long.add(x0, y0)

        # Only put into medium memory if it is a valid point
        if np.isnan(y0).any():
            self.mem_ban.add(x0, y0)
        else:
            self.mem_med.add(x0, y0)

        if self.verbose:
            print("  y = %s" % np.array_str(y0))

        return y0

    def feasible_moves(self, x0, dx):
        """Starting from x0, all moves within constraints and not tabu."""
        # Generate candidate moves
        X = hj_move(x0, dx)

        # Remove duplicate moves (can arise if an element of dx is zero)
        X = np.unique(X, axis=0)

        # Filter by input constraints
        X = X[self.constraint(X)]

        # Filter against short term memory
        X = X[~self.mem_short.contains(X)]

        # Filter against permanent ban list
        # (we put points where CFD results indicate constraint violated here)
        X = X[~self.mem_ban.contains(X)]

        return X

    def evaluate_moves(self, x0, dx):
        """From a given start point, evaluate permissible candidate moves."""

        X = self.feasible_moves(x0, dx)

        # Check which points we have seen before
        log_seen = self.mem_long.contains(X)
        X_seen = X[log_seen]
        X_unseen = X[~log_seen]

        # Re-use previous objectives from long-term mem if possible
        Y_seen = self.mem_long.lookup(X_seen)

        # Only go as far as evaluating unseen if there are actually points
        if X_unseen.shape[0] > 0:

            # Shuffle the unseen points to remove selection bias
            np.random.shuffle(X_unseen)

            # Limit the maximum parallel evaluations
            if self.max_parallel:

                # Evaluate in batches
                isplit = range(
                    self.max_parallel, len(X_unseen), self.max_parallel
                )
                X_batch = np.split(X_unseen, isplit)
                Y_batch = [self.objective(Xi) for Xi in X_batch]

                # Join results
                Y_unseen = np.concatenate(Y_batch)

            else:

                # Evaluate objective for unseen points
                Y_unseen = self.objective(X_unseen)

            # Increment function evaluation counter
            self.fevals += len(X_unseen)

            # Join the results together
            X = np.vstack((X_seen, X_unseen))
            Y = np.vstack((Y_seen, Y_unseen))

        # If there are no unseen points
        else:

            X = X_seen
            Y = Y_seen

        return X, Y

    def select_move(self, x0, y0, X, Y):
        """Choose next move given starting point and list of candidate moves."""

        j = self.j_objective

        try:
            # Categorise the candidates for next move with respect to current
            with np.errstate(invalid="ignore"):
                b_dom = (Y[:, j] < y0[:, j]).all(axis=1)
                b_non_dom = (Y[:, j] > y0[:, j]).all(axis=1)
                b_equiv = ~np.logical_and(b_dom, b_non_dom)
        except IndexError:
            print("ERROR! in select_move")
            print("Y=%s", str(Y))
            print("y0=%s", str(y0))
            print("shape Y", Y.shape)
            print("shape y0", y0.shape)
            quit()

        # Convert to indices
        i_dom = np.where(b_dom)[0]
        i_non_dom = np.where(b_non_dom)[0]
        i_equiv = np.where(b_equiv)[0]

        # Choose the next point
        if len(i_dom) > 0:
            # If we have dominating points, randomly choose from them
            np.random.shuffle(i_dom)
            x1, y1 = X[i_dom[0]], Y[i_dom[0]]
        elif len(i_equiv) > 0:
            # Randomly choose from equivalent points
            np.random.shuffle(i_equiv)
            x1, y1 = X[i_equiv[0]], Y[i_equiv[0]]
        elif len(i_non_dom) > 0:
            # Randomly choose from non-dominating points
            np.random.shuffle(i_non_dom)
            x1, y1 = X[i_non_dom[0]], Y[i_non_dom[0]]
        else:
            raise Exception("No valid points to pick next move from")

        # Keep in matrix form
        x1 = np.atleast_2d(x1)
        y1 = np.atleast_2d(y1)

        return x1, y1

    def pattern_move(self, x0, y0, x1, y1, dx):
        """If this move is in a good direction, increase move length."""
        x1a = x0 + self.fac_pattern * (x1 - x0)

        # # If we are running objectives in parallel, do not waste the spare cores
        # if self.max_parallel:

        # # Pick (n_parallel - 1) feasible moves from the pattern move point
        # X1a = self.feasible_moves(x1a, dx)
        # X1a_unseen = X1a[~self.mem_long.contains(X1a)]
        # np.random.shuffle(X1a_unseen)
        # X1a_unseen = X1a_unseen[: (self.max_parallel -1)]
        # X = np.vstack(x1,X1a_unseen)
        # Y = self.objective(X)

        # else:

        y1a = self.objective(x1a)
        if (y1a < y1).all():
            return x1a
        else:
            return x1

    def update_med(self, X, Y):
        """Update the near-optimal points in medium term memory."""

        if X.shape[0] == 0:
            flag = False
        else:
            if len(self.j_objective) == 1:
                flag = self.mem_med.update_best(X, Y)
            else:
                flag = self.mem_med.update_front(X, Y)

        return flag

    def search(self, x0, dx, callback=None):
        """Perform a search with given intial point and step size."""

        # Evaluate the objective at given initial guess point, update memories
        y0 = self.initial_guess(x0)

        max_step = dx * self.fac_restart ** 2.0

        # Main loop, until max evaluations reached or step size below tolerance
        self.i_search = 0
        while self.fevals < self.max_fevals and np.any(dx > self.tol):

            # Save in case we want to resume later
            self.dx = dx.reshape(-1).tolist()
            self.x0 = x0.reshape(-1).tolist()
            self.y0 = y0.reshape(-1).tolist()

            # Record our progress in a memory file, if specified
            if self.mem_file:
                self.save_memories(self.mem_file)

            # Plot stuff
            if self.verbose:
                self.plot_long("long.pdf")
                self.plot_opt("opt.pdf")

            # If we are given a callback, evaluate it now
            if callback:
                callback(self)

            # Evaluate objective for permissible candidate moves
            X, Y = self.evaluate_moves(x0, dx)

            # If any objectives are NaN, add to permanent ban list
            inan = np.isnan(Y).any(-1)
            Xnan = X[inan]
            self.mem_ban.add(Xnan)

            # Delete NaN from results
            X, Y = X[~inan], Y[~inan]

            # Put new results into long-term memory
            self.mem_long.add(X, Y)

            # Put Pareto-equivalent results into medium-term memory
            # Flag true if we sucessfully added a point
            flag = self.update_med(X, Y)

            if self.verbose and flag:
                print(
                    "NEW OPT\n  x = %s\n  y = %s"
                    % tuple([np.array_str(xy) for xy in self.mem_med.get(0)])
                )

            # Reset counter if we added to medium memory, otherwise increment
            self.i_search = 0 if flag else self.i_search + 1

            # Choose next point based on local search counter
            if self.i_search == self.i_restart:
                if self.verbose:
                    print("RESTART")
                # RESTART: reduce step sizes and randomly select from
                # medium-term
                dx *= self.fac_restart
                if len(self.j_objective) == 1:
                    # Pick the current optimum if scalar objective
                    x1, y1 = self.mem_med.get(0)
                else:
                    # Pick from sparse region of Pareto from if multi-objective
                    x1, y1 = self.mem_med.sample_sparse(self.x_regions)
                self.i_search = 0
            elif self.i_search in self.i_intensify or X.shape[0] == 0:
                # INTENSIFY: Select a near-optimal point if the medium memory
                # is populated
                if self.mem_med.npts > 0:
                    if self.verbose:
                        print("INTENSIFY")
                    x1, y1 = self.mem_med.sample_random()
                else:
                    # If nothing in the medium-term memory, we have not found
                    # any valid points yet, so increase step size and try again
                    if np.all(dx <= max_step):
                        if self.verbose:
                            print("INCREASE STEP")
                        dx /= self.fac_restart
                        x1, y1 = x0, y0
                    else:
                        print(
                            "Could not find a point satisfying constraints near initial guess, quitting."
                        )
            elif self.i_search == self.i_diversify:
                if self.verbose:
                    print("DIVERSIFY")
                # DIVERSIFY: Generate a new point in sparse design region
                x1 = self.mem_long.generate_sparse(self.x_regions)
                y1 = self.objective(x1)
            else:
                if self.verbose:
                    print("i=%d, fevals=%d" % (self.i_search, self.fevals))
                # Normally, choose the best candidate move
                x1, y1 = self.select_move(x0, y0, X, Y)
                # Check for a pattern move every i_pattern steps
                if not self.i_pattern is None:
                    if np.mod(self.i_search, self.i_pattern):
                        x1 = self.pattern_move(x0, y0, x1, y1, dx)

            if self.verbose:
                print(
                    "  x = %s\n  y = %s"
                    % tuple([np.array_str(xy) for xy in (x1, y1)])
                )
            # Add chosen point to short-term list (tabu)
            self.mem_short.add(x1)

            # Update current point before next iteration
            x0, y0 = x1, y1

        # After the loop return current point
        return x0, y0

    def resume(self, fname):
        self.load_memories(fname)
        self.mem_file = fname
        self.search(self.x0, self.dx)

    def save_memories(self, fname):
        """Dump the memories to a json file."""

        # Assemble a dict for each memory
        d = {k: m.to_dict() for k, m in zip(self.MEM_KEYS, self.mem_all)}

        for a in ["i_search", "x0", "y0", "dx"]:
            d[a] = getattr(self, a)

        # Write the file
        with open(fname, "w") as f:
            json.dump(d, f)

    def load_memories(self, fname):
        """Populate memories from a json file."""

        if self.verbose:
            print("READ memories from %s" % fname)
        # Load the file
        with open(fname, "r") as f:
            d = json.load(f)

        # Populate the memories
        for k, m in zip(self.MEM_KEYS, self.mem_all):
            if self.verbose:
                print("  %s: %d points" % (k, d[k]["npts"]))
            m.from_dict(d[k])

        if "i_search" in d:
            self.i_search = d["i_search"]
            self.x0 = np.atleast_2d(d["x0"])
            self.y0 = np.atleast_2d(d["y0"])
            self.dx = np.atleast_2d(d["dx"])

    def plot_long(self, fname):
        """Generate a graph of long-term memory."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        Yl = np.flip(self.mem_long.Y, axis=0) * 100.0
        pts = np.arange(len(Yl))
        Ym = self.mem_med.Y
        _, ind = find_rows(Ym, Yl)
        ax.plot(pts, Yl[:, 0], "k-")
        ax.plot(pts[ind], Yl[ind, 0], "r*")
        ax.set_ylabel("Lost Efficiency, $\Delta \eta/\%$")
        ax.set_xlabel("Design Evaluations")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def plot_opt(self, fname):
        """Generate a graph of optimisation progress."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        Yl = np.flip(self.mem_long.Y, axis=0)
        pts = np.arange(len(Yl))
        Ymin = np.empty_like(pts)
        for i, p in enumerate(pts):
            if np.all(np.isnan(Yl[: (p + 1), 0])):
                Ymin[i] = np.nan
            else:
                Ymin[i] = np.nanmin(Yl[: (p + 1), 0]) * 100.0
        ax.plot(pts, Ymin - Ymin[-1], "k-")
        ax.set_ylabel("Optimum Lost Efficiency Error, $\Delta \eta/\%$")
        ax.set_xlabel("Design Evaluations")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
