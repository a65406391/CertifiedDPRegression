from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


def find_polytope_vertices(A, b, tol=1e-10):
    """Find vertices of polytope defined by Ax <= b using vertex enumeration.

    Parameters:
    A (ndarray): (m x n) inequality matrix
    b (ndarray): (m,) inequality bounds
    tol (float): numerical tolerance

    Returns:
    vertices (ndarray): array of vertex coordinates
    """
    A = np.asarray(A)
    b = np.asarray(b).flatten()
    m, n = A.shape

    vertices = []

    # Try all combinations of n constraints (potential vertices)
    for constraint_indices in combinations(range(m), n):
        try:
            # Solve system A_subset * x = b_subset
            A_subset = A[list(constraint_indices)]
            b_subset = b[list(constraint_indices)]

            # Check if system is solvable (matrix is invertible)
            if np.linalg.matrix_rank(A_subset) < n:
                continue

            x = np.linalg.solve(A_subset, b_subset)

            # Check if point satisfies all constraints Ax <= b
            if np.all(A @ x <= b + tol):
                vertices.append(x)

        except (np.linalg.LinAlgError, ValueError):
            continue

    if len(vertices) == 0:
        # Fallback: try to find some points using optimization
        vertices = find_vertices_via_optimization(A, b)

    return np.array(vertices)


def find_vertices_via_optimization(A, b, num_directions=20):
    """Find vertices by optimizing in multiple directions."""
    A = np.asarray(A)
    b = np.asarray(b).flatten()
    m, n = A.shape

    vertices = []

    # Generate random directions and optimize
    for _ in range(num_directions):
        c = np.random.randn(n)

        # Minimize c^T x subject to Ax <= b
        result = linprog(
            c, A_ub=A, b_ub=b, bounds=(-np.inf, np.inf), method="highs"
        )
        if result.success:
            vertices.append(result.x)

        # Maximize c^T x (minimize -c^T x)
        result = linprog(
            -c, A_ub=A, b_ub=b, bounds=(-np.inf, np.inf), method="highs"
        )
        if result.success:
            vertices.append(result.x)

    # Remove duplicates
    if vertices:
        vertices = np.array(vertices)
        # Simple duplicate removal based on tolerance
        unique_vertices = []
        for v in vertices:
            is_duplicate = False
            for existing in unique_vertices:
                if np.linalg.norm(v - existing) < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_vertices.append(v)
        vertices = (
            np.array(unique_vertices) if unique_vertices else np.array(vertices)
        )

    return vertices


def project_and_plot_polytope(
    A,
    b,
    projection_matrix=None,
    ax=None,
    color="lightblue",
    edgecolor="black",
    alpha=0.5,
    method="auto",
):
    """Projects and plots a high-dimensional polytope defined by A x <= b to 2D.

    Parameters:
    A (ndarray): (m x n) inequality matrix.
    b (ndarray): (m,) inequality bounds.
    projection_matrix (ndarray): Optional (2 x n) projection matrix.
    ax (matplotlib axis): Optional matplotlib axis to plot on.
    color, edgecolor, alpha: Matplotlib styling.
    method (str): Method to use for finding vertices. Options:
        - "auto": Automatically choose best method (default)
        - "combinatorial": Use combinatorial vertex enumeration
        - "optimization": Use optimization-based vertex finding
        - "sampling": Use random sampling of feasible region
    """
    A = np.asarray(A)
    b = np.asarray(b).flatten()
    m, n = A.shape

    if projection_matrix is None:
        projection_matrix = np.eye(2, n)  # Project to first two coordinates

    # Find vertices of the polytope based on chosen method
    try:
        if method == "auto":
            # Choose method based on problem size
            if m <= 20 and n <= 6:
                vertices = find_polytope_vertices(A, b)
            else:
                vertices = find_vertices_via_optimization(A, b)
        elif method == "combinatorial":
            vertices = find_polytope_vertices(A, b)
        elif method == "optimization":
            vertices = find_vertices_via_optimization(A, b)
        elif method == "sampling":
            visualize_feasible_region_sampling(
                A, b, projection_matrix, ax, color, edgecolor, alpha
            )
            return
        else:
            raise ValueError(
                f"Unknown method: {method}. Choose from 'auto', "
                "'combinatorial', 'optimization', 'sampling'"
            )

        if len(vertices) == 0:
            raise ValueError(
                "No vertices found. Polytope might be empty or unbounded."
            )

        # Project to 2D
        projected = vertices @ projection_matrix.T

        # Handle case where we have very few points
        if len(projected) < 3:
            # Just plot the points
            if ax is None:
                fig, ax = plt.subplots()
            ax.scatter(
                projected[:, 0],
                projected[:, 1],
                color=color,
                edgecolor=edgecolor,
                s=50,
                alpha=alpha,
            )
            return

        # Compute convex hull in 2D
        hull = ConvexHull(projected)
        hull_points = projected[hull.vertices]

        # Close the polygon by adding the first point at the end
        hull_points = np.vstack([hull_points, hull_points[0]])

        # Plot
        if ax is None:
            fig, ax = plt.subplots()

        ax.fill(
            hull_points[:, 0],
            hull_points[:, 1],
            facecolor=color,
            edgecolor=edgecolor,
            alpha=alpha,
        )
        ax.plot(hull_points[:, 0], hull_points[:, 1], color=edgecolor)

    except Exception as e:
        print(f"Error in polytope projection with method '{method}': {e}")
        if method != "sampling":
            print("Falling back to sampling method...")
            visualize_feasible_region_sampling(
                A, b, projection_matrix, ax, color, edgecolor, alpha
            )


def visualize_feasible_region_sampling(
    A, b, projection_matrix, ax, color, edgecolor, alpha
):
    """Fallback method: visualize feasible region by sampling points."""
    if projection_matrix.shape[0] != 2:
        print("Sampling method only works for 2D projections")
        return

    # Create a grid of points and test feasibility
    n = A.shape[1]

    # Find rough bounds by optimization
    bounds = []
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1

        # Find min and max in each dimension
        result_min = linprog(
            c, A_ub=A, b_ub=b, bounds=(-np.inf, np.inf), method="highs"
        )
        result_max = linprog(
            -c, A_ub=A, b_ub=b, bounds=(-np.inf, np.inf), method="highs"
        )
        if result_min.success and result_max.success:
            min_val = result_min.fun  # This is the minimum value of x[i]
            max_val = -result_max.fun  # This is the maximum value of x[i]
            bounds.append((min_val, max_val))
        else:
            raise ValueError(
                f"Could not find bounds for dimension {i}. "
                "Polytope might be unbounded or infeasible."
            )
            bounds.append((-10, 10))  # Default bounds

    print(bounds)
    # Sample points and project
    feasible_points = []
    for _ in range(1000):
        x = np.array([np.random.uniform(low, high) for low, high in bounds])
        if np.all(A @ x <= b + 1e-10):
            feasible_points.append(x)

    if feasible_points:
        feasible_points = np.array(feasible_points)
        projected = feasible_points @ projection_matrix.T

        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            projected[:, 0], projected[:, 1], color=color, alpha=alpha, s=1
        )
