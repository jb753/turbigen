"""Functions for multiobjective tabu search."""
import numpy as np


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
        self.tol = np.array(tol)

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
        return self._X[ind], self._Y[ind]

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

        # Join memory and test points into one matrix
        Yall = np.concatenate((self.Y, Y), axis=0)
        Xall = np.concatenate((self.X, X), axis=0)

        # Sort by objective, truncate to maximum number of points
        isort = np.argsort(Yall[:, 0], axis=0)[: self.max_points]
        Xall, Yall = Xall[isort], Yall[isort]

        # Reassign to the memory
        self.npts = len(isort)
        self._X[: self.npts] = Xall
        self._Y[: self.npts] = Yall

        return np.any(self.contains(X))

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
        i_select = np.random.choice(self.npts, 1)
        return self._X[i_select], self._Y[i_select]

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


class TabuSearch:
    def __init__(self, objective, constraint, nx, ny, tol):
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
        self.n_med = 2000 if ny > 1 else 10

        # Default iteration counters
        self.i_diversify = 10
        self.i_intensify = 20
        self.i_restart = 40
        self.i_pattern = 2

        # Misc algorithm parameters
        self.x_regions = 3
        self.max_fevals = 20000
        self.fac_restart = 0.5
        self.fac_pattern = 2.0
        self.max_parallel = None

        # Initialise counters
        self.fevals = 0

        # Initialise memories
        self.mem_short = Memory(nx, ny, self.n_short, self.tol)
        self.mem_med = Memory(nx, ny, self.n_med, self.tol)
        self.mem_long = Memory(nx, ny, self.n_long, self.tol)
        self.mem_all = (self.mem_short, self.mem_med, self.mem_long)

    def clear_memories(self):
        """Erase all memories"""
        for mem in self.mem_all:
            mem.clear()

    def initial_guess(self, x0):
        """Reset memories, set current point, evaluate objective."""
        self.clear_memories()
        y0 = self.objective(x0)
        self.fevals += 1
        for mem in self.mem_all:
            mem.add(x0, y0)
        return y0

    def evaluate_moves(self, x0, dx):
        """From a given start point, evaluate permissible candidate moves."""

        # Generate candidate moves
        X = hj_move(x0, dx)

        # Filter by input constraints
        X = X[self.constraint(X)]

        # Filter against short term memory
        X = X[~self.mem_short.contains(X)]

        # Check which points we have seen before
        log_seen = self.mem_long.contains(X)
        X_seen = X[log_seen]
        X_unseen = X[~log_seen]

        # Re-use previous objectives from long-term mem if possible
        Y_seen = self.mem_long.lookup(X_seen)

        # Limit the maximum parallel evaluations
        if self.max_parallel:
            np.random.shuffle(X_unseen)
            X_unseen = X_unseen[: self.max_parallel]

        # Evaluate objective for unseen points
        Y_unseen = self.objective(X_unseen)

        # Increment function evaluation counter
        self.fevals += len(X_unseen)

        # Join the results together
        X = np.vstack((X_seen, X_unseen))
        Y = np.vstack((Y_seen, Y_unseen))

        return X, Y

    def select_move(self, x0, y0, X, Y):
        """Choose next move given starting point and list of candidate moves."""

        # Categorise the candidates for next move with respect to current
        b_dom = (Y < y0).all(axis=1)
        b_non_dom = (Y > y0).all(axis=1)
        b_equiv = ~np.logical_and(b_dom, b_non_dom)

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

    def pattern_move(self, x0, y0, x1, y1):
        """If this move is in a good direction, increase move length."""
        x1a = x0 + self.fac_pattern * (x1 - x0)
        y1a = self.objective(x1a)
        if (y1a < y1).all():
            return x1a
        else:
            return x1

    def search(self, x0, dx, callback=None):
        """Perform a search with given intial point and step size."""

        # Evaluate the objective at given initial guess point, update memories
        y0 = self.initial_guess(x0)

        # Main loop, until max evaluations reached or step size below tolerance
        i = 0
        while self.fevals < self.max_fevals and np.any(dx > self.tol):

            # If we are given a callback, evaluate it now
            if callback:
                callback(self)

            # Evaluate objective for permissible candidate moves
            X, Y = self.evaluate_moves(x0, dx)

            # If any objectives are NaN, add to tabu list
            inan = np.isnan(Y).any(-1)
            Xnan = X[inan]
            self.mem_short.add(Xnan)

            # Delete NaN from results
            X, Y = X[~inan], Y[~inan]

            # Put new results into long-term memory
            self.mem_long.add(X, Y)

            # Put Pareto-equivalent results into medium-term memory
            # Flag true if we sucessfully added a point
            if self.ny == 1:
                flag = self.mem_med.update_best(X, Y)
            else:
                flag = self.mem_med.update_front(X, Y)
            print('BEST %s %s' % self.mem_med.get(0))

            # Reset counter if we added to medium memory, otherwise increment
            i = 0 if flag else i + 1

            # Choose next point based on local search counter
            if i == self.i_restart:
                print('*RESTART*')
                # RESTART: reduce step sizes and randomly select from
                # medium-term
                dx = dx * self.fac_restart
                if self.ny == 1:
                    # Pick the current optimum if scalar objective
                    x1, y1 = self.mem_med.get(0)
                else:
                    # Pick from sparse region of Pareto from if multi-objective
                    x1, y1 = self.mem_med.sample_sparse(self.x_regions)
                i = 0
            elif i == self.i_intensify or X.shape[0] == 0:
                print('*INTENSIFY*')
                # INTENSIFY: Select a near-optimal point
                x1, y1 = self.mem_med.sample_sparse(self.x_regions)
            elif i == self.i_diversify:
                print('*DIVERSIFY*')
                # DIVERSIFY: Generate a new point in sparse design region
                x1 = self.mem_long.generate_sparse(self.x_regions)
                y1 = self.objective(x1)
            else:
                # Normally, choose the best candidate move
                x1, y1 = self.select_move(x0, y0, X, Y)
                # Check for a pattern move every i_pattern steps
                if np.mod(i, self.i_pattern):
                    x1 = self.pattern_move(x0, y0, x1, y1)

            # Add chosen point to short-term list (tabu)
            self.mem_short.add(x1)

            # Update current point before next iteration
            x0, y0 = x1, y1

        # After the loop return current point
        return x0, y0
