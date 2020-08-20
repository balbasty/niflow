import unik as U
import unik.symbolik as US


@U.tensor_compat(map_batch=False)
def meanm(mats, max_iter=1024, tol=1e-20):
    """Compute the exponential barycentre of a set of matrices.

    Parameters
    ----------
    mats : (N, M, M) tensor_like
        Set of square invertible matrices
    max_iter : int, default=1024
        Maximum number of iterations
    tol : float, default=1E-20
        Tolerance for early stopping.
        The tolerance criterion is the sum-of-squares of the residuals
        in log-space, _i.e._, :math:`||\sum_n \log_M(A_n) / N||^2`

    Returns
    -------
    mean_mat : (M, M) tensor or array
        Mean matrix.

    References
    ----------
    .. [1]  Xavier Pennec, Vincent Arsigny.
        "Exponential Barycenters of the Canonical Cartan Connection and
        Invariant Means on Lie Groups."
        Matrix Information Geometry, Springer, pp.123-168, 2012.
        (10.1007/978-3-642-30232-9_7). (hal-00699361)
        https://hal.inria.fr/hal-00699361

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Tensorflow port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    mats = U.as_tensor(mats)
    dim = U.shape(mats)[-1] - 1
    in_dtype = U.dtype(mats)
    acc_dtype = 'float64'

    def body(mean_mat, sos):
        # Project all matrices to the tangent space about the current mean_mat
        log_mats = U.map_fn(lambda x: U.logm(U.lmdiv(mean_mat, x)), mats)
        # Compute the new mean in the tangent space
        mean_log_mat = U.mean(log_mats, axis=0)
        # Compute sum-of-squares in tangent space (should be zero at optimum)
        sos = U.sum(mean_log_mat ** 2)
        # Exponentiate to original space
        mean_mat = U.matmul(mean_mat, U.expm(mean_log_mat))
        return mean_mat, sos

    def cond(mean_mat, sos):
        return sos >= tol

    mats = U.cast(mats, acc_dtype)
    mean_mat = U.eye(dim+1, dtype=acc_dtype)
    mean_mat, _ = U.while_loop(cond, body, [mean_mat, U.inf],
                               maximum_iterations=max_iter)

    return U.cast(mean_mat, in_dtype)


@U.tensor_compat(return_dtype=[US.dtype(US.Arg('X')), US.dtype(US.Arg('X'))])
def dexpm(X, basis, max_order=10000, tol=1e-32):
    """Derivative of the matrix exponential.

    This function evaluates the matrix exponential and its derivative
    using a Taylor approximation. A faster integration technique, based
    e.g. on scaling and squaring, could have been used instead.

    Parameters
    ----------
    X : {(F,), (D, D)} tensor_like
        If vector_like: parameters of the log-matrix in the basis set
        If matrix_like: log-matrix
    basis : (F, D, D) tensor_like
        Basis set
    max_order : int, default=10000
        Order of the Taylor expansion
    tol : float, default=1e-32
        Tolerance for early stopping
        The criterion is based on the Frobenius norm of the last term of
        the Taylor series.

    Returns
    -------
    eX : (D, D) tensor or  array
        Matrix exponential
    dX : (F, D, D) tensor or  array
        Derivative of the matrix exponential with respect to the
        parameters in the basis set

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python/Unik port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    # TODO: should I do the computations with the type of X, instead
    #       of casting everything to double?

    X = U.as_tensor(X)
    in_dtype = U.dtype(X)
    X = U.cast(X, 'float64')
    if basis is None:
        return None
    basis = U.as_tensor(basis, dtype='float64')

    # If X is a vector: assume that it contains parameters in the algebra
    #                   -> reconstruct matrix
    X = U.cond(U.rank(X) == 1,
               lambda: U.sum(basis * X[:, None, None], axis=0),
               lambda: X)

    # Aliases
    I = U.eye(U.length(X), dtype='float64')
    E = I + X                            # expm(X)
    dE = U.copy(basis)                   # dexpm(X)
    En = U.copy(X)                       # n-th Taylor coefficient of expm
    dEn = U.copy(basis)                  # n-th Taylor coefficient of dexpm

    def body(E, En, dE, dEn, n_order):
        dEn = (U.matmul(dEn, X[None, ...]) + U.matmul(En[None, ...], basis))
        dEn /= n_order
        dE += dEn
        En = U.matmul(En, X)/n_order
        E += En
        n_order += 1
        return E, En, dE, dEn, n_order

    def cond(E, En, dE, dEn, n_order):
        sos_cond = U.sum(En ** 2) >= U.size(En) * tol
        iter_cond = n_order <= max_order+1
        return sos_cond & iter_cond

    E, En, dE, dEn, n_order = U.while_loop(cond, body, [E, En, dE, dEn, 2])

    E = U.cast(E, in_dtype)
    dE = U.cast(dE, in_dtype)
    return E, dE
