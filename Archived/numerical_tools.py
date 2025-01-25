"""Write structure for computational methods."""

TOL = 0.001


def deriv(f, x, del_x, prev=1):
    """Numerical approximation for function of one variable."""
    frac = (f(x + del_x) - f(x - del_x)) / (2 * del_x)
    if abs(frac - prev) < TOL:
        return frac
    return deriv(f, x, del_x / 10, frac)
