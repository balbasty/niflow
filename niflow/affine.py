# Python
from copy import deepcopy
from ast import literal_eval
# Numpy
import numpy as np
# Unik
import unik
from unik import symbolik


@unik.tensor_compat
def volume_axis(*args, **kwargs):
    """Describe an axis of a volume of voxels.

    Signature
    ---------
    volume_axis(index, flipped=False)
    volume_axis(name)
    volume_axis(axis)

    Parameters
    ----------
    index : () tensor_like[int]
        Index of the axis in 'direct' space (RAS)

    flipped : () tensor_like[bool], default=False
        Whether the axis is flipped or not.

    name : {'R' or 'L', 'A' or 'P', 'S' or 'I'}
        Name of the axis, according to the neuroimaging convention:
        * 'R' for *left to Right* (index=0, flipped=False) or
          'L' for *right to Left* (index=0, flipped=True)
        * 'A' for *posterior to Anterior* (index=1, flipped=False) or
          'P' for *anterior to Posterior* (index=1, flipped=True)
        * 'S' for *inferior to Superior* (index=2, flipped=False) or
          'I' for *superior to Inferior* (index=2, flipped=True)

    axis : (2,) tensor_like[int]
        ax[0] = index
        ax[1] = flipped

    Returns
    -------
    ax : (2,) tensor[int]
        Description of the axis.

    """
    def axis_from_name(name):
        name = name.upper()
        if name == 'R':
            return np.array([0, 0], dtype='int')
        elif name == 'L':
            return np.array([0, 1], dtype='int')
        elif name == 'A':
            return np.array([1, 0], dtype='int')
        elif name == 'P':
            return np.array([1, 1], dtype='int')
        elif name == 'S':
            return np.array([2, 0], dtype='int')
        elif name == 'I':
            return np.array([2, 1], dtype='int')

    def axis_from_index(index, flipped=False):
        return unik.cast(unik.as_tensor([index, flipped]), 'int')

    def axis_from_axis(ax):
        unik.assert_equal(unik.ndim(ax), 2)
        return unik.cast(unik.as_tensor(ax), 'int')

    # Dispatch based on input types or keyword arguments
    args = list(args)
    if len(args) > 0:
        if isinstance(args[0], str):
            return axis_from_name(*args, **kwargs)
        else:
            args[0] = unik.as_tensor(args[0])
            return unik.cond(unik.ndim(args) == 0,
                               true_fn=axis_from_index(*args, **kwargs),
                               false_fn=axis_from_axis(*args, **kwargs))
    else:
        if 'name' in kwargs.keys():
            return axis_from_name(*args, **kwargs)
        elif 'index' in kwargs.keys():
            return axis_from_index(*args, **kwargs)
        else:
            return axis_from_axis(*args, **kwargs)


@unik.tensor_compat
def volume_layout(*args, **kwargs):
    """Describe the layout of a volume of voxels.

    A layout is characterized by a list of axes. See `axis`.

    Signature
    ---------
    volume_layout(name='RAS')
    volume_layout(axes)
    volume_layout(index, flipped=False)

    Parameters
    ----------
    name : str, default='RAS'
        Permutation of axis names,  according to the neuroimaging convention:
        * 'R' for *left to Right* (index=0, flipped=False) or
          'L' for *right to Left* (index=0, flipped=True)
        * 'A' for *posterior to Anterior* (index=1, flipped=False) or
          'P' for *anterior to Posterior* (index=1, flipped=True)
        * 'S' for *inferior to Superior* (index=2, flipped=False) or
          'I' for *superior to Inferior* (index=2, flipped=True)
        The number of letters defines the dimension of the matrix
        (`ndim = len(name)`).

    axes : (ndim, 2) tensor_like[int]
        List of objects returned by `axis`.

    index : (ndim, ) tensor_like[int]
        Index of the axess in 'direct' space (RAS)

    flipped : (ndim, ) tensor_like[bool], default=False
        Whether each axis is flipped or not.

    Returns
    -------
    layout : (ndim, 2) tensor[int]
        Description of the layout.

    """
    def layout_from_name(name):
        return volume_layout([volume_axis(a) for a in name])

    def layout_from_index(index, flipped=False):
        index = unik.as_tensor(index)
        ndim = index.shape[0]
        flipped = unik.as_tensor(flipped)
        flipped = unik.cond(unik.rank(flipped) == 0,
                            lambda: flipped[None, ...],
                            lambda: flipped)
        flipped = unik.cond(unik.shape(flipped)[0] == 1,
                            lambda: unik.tile(flipped, [ndim]),
                            lambda: flipped)
        flipped = unik.cast(flipped, 'int')
        return unik.cast(unik.stack((index, flipped)), 'int')

    def layout_from_axes(axes):
        unik.assert_equal(unik.ndim(axes), 2)
        return unik.cast(axes, 'int')

    # Dispatch based on input types or keyword arguments
    args = list(args)
    if len(args) > 0:
        if isinstance(args[0], str):
            return layout_from_name(*args, **kwargs)
        else:
            args[0] = unik.as_tensor(args[0])
            return unik.cond(unik.ndim(args[0]) == 1,
                             true_fn=lambda: layout_from_index(*args, **kwargs),
                             false_fn=lambda: layout_from_axes(*args, **kwargs))
    else:
        if 'name' in kwargs.keys():
            return layout_from_name(*args, **kwargs)
        elif 'index' in kwargs.keys():
            return layout_from_index(*args, **kwargs)
        else:
            return layout_from_axes(*args, **kwargs)


@unik.tensor_compat
def iter_layout(ndim):
    """Compute all possible layouts for a given dimensionality.

    Parameters
    ----------
    ndim : () tensor_like
        Dimensionality (rank) of the space.

    Returns
    -------
    layouts : (F*P, ndim, 2) tensor[int]
        All possible layouts.
        * F = 2 ** ndim     -> number of flips
        * P = ndim!         -> number of permutations

    """
    # First, compute all possible directed layouts on one hand,
    # and all possible flips on the other hand.
    axes = unik.range(ndim)
    layouts = unik.permutations(axes)                 # [P, D]
    flips = unik.cartesian([0, 1], shape=ndim)        # [F, D]

    # Now, compute combination (= cartesian product) of both
    # We replicate each tensor so that shapes match and stack them.
    nb_layouts = unik.length(layouts)
    nb_flips = unik.length(flips)
    layouts = layouts[None, ...]
    layouts = unik.tile(layouts, [nb_flips, 1, 1])    # [F, P, D]
    flips = flips[:, None, :]
    flips = unik.tile(flips, [1, nb_layouts, 1])      # [F, P, D]
    layouts = unik.stack([layouts, flips], axis=-1)

    # Finally, flatten across repeats
    layouts = unik.reshape(layouts, [-1, ndim, 2])    # [F*P, D, 2]

    return layouts


@unik.tensor_compat
def layout_matrix(layout, dtype='float64'):
    """Compute the origin affine matrix for different voxel layouts.

    Resources
    ---------
    .. https://nipy.org/nibabel/image_orientation.html
    .. https://nipy.org/nibabel/neuro_radio_conventions.html

    Parameters
    ----------
    layout : str or (ndim, 2) tensor_like[int]
        See `affine.layout`

    dtype : str or type
        Data type of the matrix

    Returns
    -------
    mat : (ndim+1, ndim+1) tensor[dtype]
        Corresponding affine matrix.

    """
    # TODO: shape argument to include the translation induced by flips.

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Extract info from layout
    layout = volume_layout(layout)
    dim = unik.length(layout)
    perm = unik.invert_permutation(layout[:, 0])
    flip = unik.cast(layout[:, 1], 'bool')

    # Create matrix
    mat = unik.eye(dim+1, dtype=dtype)
    mat = mat[perm + [3], :]
    mflip = unik.ones(dim+1, dtype=dtype)
    mflip = unik.where(unik.concat((flip, [False])), -mflip, mflip)
    mflip = unik.diag(mflip)
    mat = unik.matmul(mflip, mat)

    return mat


@unik.tensor_compat
def affine_to_layout(mat):
    """Find the volume layout associated with an affine matrix.

    Parameters
    ----------
    mat : (dim, dim+1) or (dim+1, dim+1) tensor_like
        Affine matrix

    Returns
    -------
    layout : (dim, 2) tensor
        Volume layout (see `volume_layout`)

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original idea
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    # Extract linear component + remove voxel scaling
    mat = unik.as_tensor(mat)
    dtype = unik.dtype(mat)
    mat = unik.cast(mat, dtype='float64')
    dim = unik.shape(mat)[-1] - 1
    mat = mat[:dim, :dim]
    vs = unik.sum(mat ** 2, axis=1)
    mat = unik.rmdiv(mat, np.diag(vs))
    eye = unik.eye(dim, dtype='float64')

    # Compute SOS between a layout matrix and the (stripped) affine matrix
    def check_space(space):
        layout = layout_matrix(space)[:dim, :dim]
        sos = unik.sum((unik.rmdiv(mat, layout) - eye) ** 2)
        return sos

    # Compute SOS between each layout and the (stripped) affine matrix
    all_layouts = iter_layout(dim)
    all_sos = unik.map_fn(check_space, all_layouts)
    argmin_layout = unik.argmin(all_sos)
    min_layout = all_layouts[argmin_layout, ...]

    return unik.cast(min_layout, dtype)


affine_subbasis_choices = ('T', 'R', 'Z', 'Z0', 'I', 'S')


@unik.tensor_compat(return_dtype=unik.Arg('dtype'))
def affine_subbasis(mode, dim=3, sub=None, dtype='float64'):
    """Generate a basis set for the algebra of some (Lie) group of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group required does not require translations. To extract the linear
    part of the basis: lin = basis[:-1, :-1].

    This function focuses on very simple (and coherent) groups.

    Parameters
    ----------
    mode : {'T', 'R', 'Z', 'Z0', 'I', 'S'}
        Group that should be encoded by the basis set:
            * 'T'   : Translations                     [dim]
            * 'R'   : Rotations                        [dim*(dim-1)//2]
            * 'Z'   : Zooms (= anisotropic scalings)   [dim]
            * 'Z0'  : Isovolumic scalings              [dim-1]
            * 'I'   : Isotropic scalings               [1]
            * 'S'   : Shears                           [dim*(dim-1)//2]
        If the group name is appended with a list of integers, they
        have the same use as ``sub``. For example 'R[0]' returns the
        first rotation basis only. This grammar cannot be used in
        conjunction with the ``sub`` keyword.

    dim : {1, 2, 3}, default=3
        Dimension

    sub : int or list[int], optional
        Request only subcomponents of the basis

    dtype : str or type, default='float64'
        Data type of the returned array

    Returns
    -------
    basis : (F, dim+1, dim+1) tensor[dtype]
        Basis set, where ``F`` is the number of basis functions.

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    # Check if sub passed in mode
    mode = mode.split('[')
    if len(mode) > 1:
        if sub is not None:
            raise ValueError('Cannot use both ``mode`` and ``sub`` '
                             'to specify a sub-basis.')
        sub = '[' + mode[1]
        sub = literal_eval(sub)  # Safe eval for list of native types
    mode = mode[0]

    dim = unik.reshape(dim, ())
    if not unik.is_tensor(dim, 'tf') and dim not in (1, 2, 3):
        raise ValueError('dim must be one of 1, 2, 3')
    if mode not in affine_subbasis_choices:
        raise ValueError('mode must be one of {}.'
                         .format(affine_subbasis_choices))

    # Compute the basis

    if mode == 'T':
        basis = unik.zeros((dim, dim+1, dim+1), dtype=dtype)
        def body(basis, i):
            return unik.scatter([[i, i, dim]], [1],
                                basis, mode='update'), i+1
        def cond(_, i): return i < dim
        basis = unik.while_loop(cond, body, [basis, 0])[0]

    elif mode == 'Z':
        basis = unik.zeros((dim, dim+1, dim+1), dtype=dtype)
        def body(basis, i):
            return unik.scatter([[i, i, i]], [1],
                                basis, mode='update'), i+1
        def cond(_, i): return i < dim
        basis = unik.while_loop(cond, body, [basis, 0])[0]

    elif mode == 'Z0':
        basis = unik.zeros((dim-1, dim+1), dtype=dtype)
        def body(basis, i):
            return unik.scatter([[i, i], [i, i+1]], [1, -1],
                                basis, mode='update'), i+1
        def cond(_, i): return i < dim-1
        basis = unik.while_loop(cond, body, [basis, 0])[0]
        # Orthogonalise numerically (is there an analytical form?)
        u, s, v = unik.svd(basis)
        basis = unik.mm(unik.transpose(u), basis)
        basis = unik.mm(basis, v)
        basis = unik.lmdiv(unik.diag(s), basis)
        basis = unik.map_fn(unik.diag, basis)

    elif mode == 'I':
        basis = unik.zeros((1, dim+1, dim+1), dtype=dtype)
        def body(basis, i):
            return unik.scatter([[0, i, i]], [1], basis, mode='update'), i+1
        def cond(_, i): return i < dim
        basis = unik.while_loop(cond, body, [basis, 0])[0]

    elif mode == 'R':
        basis = unik.zeros((dim*(dim-1)//2, dim+1, dim+1), dtype=dtype)
        def body(basis, i, j, k):
            ind = [[k, i, j], [k, j, i]]
            val = [1/np.sqrt(2), -1/np.sqrt(2)]
            basis = unik.scatter(ind, val, basis, mode='update')
            j = unik.cond(j+1 < dim, lambda: j+1, lambda: 0)
            i = unik.cond(j == 0, lambda: i+1, lambda: i)
            j = unik.cond(j == 0, lambda: i+1, lambda: j)
            k = k + 1
            return basis, i, j, k
        def cond(_, i, j, k): return (i < dim) & (j < dim)
        basis = unik.while_loop(cond, body, [basis, 0, 1, 0])[0]

    elif mode == 'S':
        basis = unik.zeros((dim*(dim-1)//2, dim+1, dim+1), dtype=dtype)
        def body(basis, i, j, k):
            ind = [[k, i, j], [k, j, i]]
            val = [1/np.sqrt(2), 1/np.sqrt(2)]
            basis = unik.scatter(ind, val, basis, mode='update')
            j = unik.cond(j+1 < dim, lambda: j+1, lambda: 0)
            i = unik.cond(j == 0, lambda: i+1, lambda: i)
            j = unik.cond(j == 0, lambda: i+1, lambda: j)
            k = k + 1
            return basis, i, j, k
        def cond(_, i, j, k): return (i < dim) & (j < dim)
        basis = unik.while_loop(cond, body, [basis, 0, 1, 0])[0]

    else:
        # We should never reach this (a test was performed earlier)
        raise ValueError

    # Select subcomponents of the basis
    if sub is not None:
        try:
            sub = list(sub)
        except TypeError:
            sub = [sub]
        basis = unik.stack([basis[i, ...] for i in sub])

    return unik.cast(basis, dtype)


affine_basis_choices = ('T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+')


@unik.tensor_compat(return_dtype=unik.Arg('dtype'))
def affine_basis(group='SE', dim=3, dtype='float64'):
    """Generate a basis set for the algebra of some (Lie) group of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group does not require translations. To extract the linear
    part of the basis: lin = basis[:-1, :-1].

    This function focuses on 'classic' Lie groups. Note that, while it
    is commonly used in registration software, we do not have a
    "9-parameter affine" (translations + rotations + zooms),
    because such transforms do not form a group; that is, their inverse
    may contain shears.

    Parameters
    ----------
    group : {'T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+'}, default='SE'
        Group that should be encoded by the basis set:
            * 'T'   : Translations
            * 'SO'  : Special Orthogonal (rotations)
            * 'SE'  : Special Euclidean (translations + rotations)
            * 'D'   : Dilations (translations + isotropic scalings)
            * 'CSO' : Conformal Special Orthogonal
                      (translations + rotations + isotropic scalings)
            * 'SL'  : Special Linear (rotations + isovolumic zooms + shears)
            * 'GL+' : General Linear [det>0] (rotations + zooms + shears)
            * 'Aff+': Affine [det>0] (translations + rotations + zooms + shears)
    dim : {1, 2, 3}, default=3
        Dimension
    dtype : str or type, default='float64'
        Data type of the returned array

    Returns
    -------
    basis : (F, dim+1, dim+1) tensor[dtype]
        Basis set, where ``F`` is the number of basis functions.

    """
    # TODO:
    # - other groups?

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    if not unik.is_tensor(dim, 'tf') and dim not in (1, 2, 3):
        raise ValueError('dim must be one of 1, 2, 3')
    if group not in affine_basis_choices:
        raise ValueError('group must be one of {}.'
                         .format(affine_basis_choices))

    if group == 'T':
        return affine_subbasis('T', dim, dtype=dtype)
    elif group == 'SO':
        return affine_subbasis('R', dim, dtype=dtype)
    elif group == 'SE':
        return unik.concat((affine_subbasis('T', dim, dtype=dtype),
                            affine_subbasis('R', dim, dtype=dtype)))
    elif group == 'D':
        return unik.concat((affine_subbasis('T', dim, dtype=dtype),
                            affine_subbasis('I', dim, dtype=dtype)))
    elif group == 'CSO':
        return unik.concat((affine_subbasis('T', dim, dtype=dtype),
                            affine_subbasis('R', dim, dtype=dtype),
                            affine_subbasis('I', dim, dtype=dtype)))
    elif group == 'SL':
        return unik.concat((affine_subbasis('R', dim, dtype=dtype),
                            affine_subbasis('Z0', dim, dtype=dtype),
                            affine_subbasis('S', dim, dtype=dtype)))
    elif group == 'GL+':
        return unik.concat((affine_subbasis('R', dim, dtype=dtype),
                            affine_subbasis('Z', dim, dtype=dtype),
                            affine_subbasis('S', dim, dtype=dtype)))
    elif group == 'Aff+':
        return unik.concat((affine_subbasis('T', dim, dtype=dtype),
                            affine_subbasis('R', dim, dtype=dtype),
                            affine_subbasis('Z', dim, dtype=dtype),
                            affine_subbasis('S', dim, dtype=dtype)))


def _format_basis(basis, dim=None):
    """Transform an Outter/Inner Lie basis into a list of arrays."""

    basis = deepcopy(basis)
    if isinstance(basis, str):
        basis = [basis]

    # Guess dimension
    if dim is None:
        if unik.is_tensor(basis):
            dim = unik.shape(basis)[-1] - 1
        else:
            for outer_basis in basis:
                if unik.is_tensor(outer_basis):
                    dim = unik.shape(outer_basis)[0] - 1
                    break
                elif not isinstance(outer_basis, str):
                    for inner_basis in outer_basis:
                        if not isinstance(inner_basis, str):
                            inner_basis = unik.as_tensor(inner_basis)
                            dim = unik.shape(inner_basis)[0] - 1
                            break
    if dim is None:
        # Guess failed
        dim = 3

    # Helper to convert named bases to matrices
    def name_to_basis(name):
        basename = name.split('[')[0]
        if basename in affine_subbasis_choices:
            return affine_subbasis(name, dim)
        elif basename in affine_basis_choices:
            return affine_basis(name, dim)
        else:
            raise ValueError('Unknown basis name {}.'
                             .format(name))

    # Convert 'named' bases to matrix bases
    if not unik.is_tensor(basis):
        basis = list(basis)
        for n_outer, outer_basis in enumerate(basis):
            if isinstance(outer_basis, str):
                basis[n_outer] = name_to_basis(outer_basis)
            elif not unik.is_tensor(outer_basis):
                outer_basis = list(outer_basis)
                for n_inner, inner_basis in enumerate(outer_basis):
                    if isinstance(inner_basis, str):
                        outer_basis[n_inner] = name_to_basis(inner_basis)
                    else:
                        outer_basis[n_inner] = unik.as_tensor(inner_basis)
                outer_basis = unik.concat(outer_basis)
                basis[n_outer] = outer_basis

    return basis, dim


@unik.tensor_compat(return_dtype=symbolik.dtype(symbolik.Arg('prm')))
def affine_matrix(prm, basis, dim=None, layout=None):
    r"""Reconstruct an affine matrix from its Lie parameters.

    Affine matrices are encoded as product of sub-matrices, where
    each sub-matrix is encoded in a Lie algebra. Finally, the right
    most matrix is a 'layout' matrix (see affine_layout).
    ..math: M   = exp(A_1) \times ... \times exp(A_n) \times L
    ..math: A_i = \sum_k = p_{ik} B_{ik}

    An SPM-like construction (as in ``spm_matrix``) would be:
    >>> M = affine_matrix(prm, ['T', 'R[0]', 'R[1]', 'R[2]', 'Z', 'S'])
    Rotations need to be split by axis because they do not commute.

    Parameters
    ----------
    prm : vector_like or vector_like[vector_like]
        Parameters in the Lie algebra(s).

    basis : vector_like[basis_like]
        The outer level corresponds to matrices in the product (*i.e.*,
        exponentiated matrices), while the inner level corresponds to
        Lie algebras.

    dim : int, default=guess or 3
        If not provided, the function tries to guess it from the shape
        of the basis matrices. If the dimension cannot be guessed
        (because all bases are named bases), the default is 3.

    layout : str or matrix_like, default=None
        A layout matrix.

    Returns
    -------
    mat : (dim+1, dim+1) tensor
        Reconstructed affine matrix.

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Make sure basis is a vector_like of (F, D+1, D+1) tensor_like
    basis, dim = _format_basis(basis, dim)

    # Check length
    nb_basis = unik.sum([unik.length(b) for b in basis])
    prm = unik.flatten(prm)
    dtype = unik.dtype(prm)
    unik.assert_equal(unik.length(prm), nb_basis, message=
                      'Number of parameters and number of bases do not match. '
                      'Got {} and {}'.format(unik.length(prm), nb_basis))

    # Helper to reconstruct a log-matrix
    def recon(p, B):
        p = unik.as_tensor(p, dtype=dtype)
        B = unik.as_tensor(B, dtype=dtype)
        return unik.expm(unik.sum(B*p[:, None, None], axis=0))

    # Reconstruct each sub matrix
    n_prm = 0
    mats = []
    for a_basis in basis:
        nb_prm = unik.length(a_basis)
        a_prm = prm[n_prm:(n_prm+nb_prm)]
        mats.append(recon(a_prm, a_basis))
        n_prm += nb_prm

    # Add layout matrix
    if layout is not None:
        layout = layout_matrix(layout)
        mats.append(layout)

    # Matrix product
    return unik.matmul_iter(mats)


@unik.tensor_compat
def affine_matrix_classic(prm, dim=3, layout=None):
    """Build an affine matrix in the "classic" way (no Lie algebra).

    Parameters
    ----------
    prm : (K,) vector_like
        Affine parameters, ordered as
        `[*translations, *rotations, *zooms, *shears]`
        Rotation parameters should be expressed in radians.
    dim : () tensor_like[int]
        Dimensionality.
    layout : str or matrix_like, default=None
        Volume layout.

    Returns
    -------
    mat : (dim+1, dim+1) tensor
        Reconstructed affine matrix `mat = T @ Rx @ Ry @ Rz @ Z @ S`

    """

    def affine_2d(t, r, z, s, dtype):
        if t is not None:
            T = [[1, 0, t[0]],
                 [0, 1, t[1]],
                 [0, 0, 1]]
        else:
            T = unik.eye(3, dtype=dtype)
        if r is not None:
            R = [[unik.cos(r[0]), unik.sin(r[0]), 0],
                 [-unik.sin(r[0]), unik.cos(r[0]), 0],
                 [0, 0, 1]]
        else:
            R = unik.eye(3, dtype=dtype)
        if z is not None:
            Z = [[z[0], 0, 0],
                 [0, z[1], 0],
                 [0, 0, 1]]
        else:
            Z = unik.eye(3, dtype=dtype)
        if s is not None:
            S = [[1, s[0], 0],
                 [0, 1, 0],
                 [0, 0, 1]]
        else:
            S = unik.eye(3, dtype=dtype)

        T = unik.as_tensor(T, dtype=dtype)
        R = unik.as_tensor(R, dtype=dtype)
        Z = unik.as_tensor(Z, dtype=dtype)
        S = unik.as_tensor(S, dtype=dtype)
        return unik.mm(T, unik.mm(R, unik.mm(Z, S)))

    def affine_3d(t, r, z, s, dtype):
        if t is not None:
            T = [[1, 0, 0, t[0]],
                 [0, 1, 0, t[1]],
                 [0, 0, 1, t[2]],
                 [0, 0, 0, 1]]
        else:
            T = unik.eye(4, dtype=dtype)
        if r is not None:
            Rx = [[1, 0, 0, 0],
                  [0, unik.cos(r[0]), unik.sin(r[0]), 0],
                  [0, -unik.sin(r[0]), unik.cos(r[0]), 0],
                  [0, 0, 0, 1]]
            Ry = [[unik.cos(r[1]), 0, unik.sin(r[1]), 0],
                  [0, 1, 0, 0],
                  [-unik.sin(r[1]), 0, unik.cos(r[1]), 0],
                  [0, 0, 0, 1]]
            Rz = [[unik.cos(r[2]), unik.sin(r[2]), 0, 0],
                  [-unik.sin(r[2]), unik.cos(r[2]), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
            Rx = unik.as_tensor(Rx, dtype=dtype)
            Ry = unik.as_tensor(Ry, dtype=dtype)
            Rz = unik.as_tensor(Rz, dtype=dtype)
            R = unik.mm(Rx, unik.mm(Ry, Rz))
        else:
            R = unik.eye(4, dtype=dtype)
        if z is not None:
            Z = [[z[0], 0, 0, 0],
                 [0, z[1], 0, 0],
                 [0, 0, z[2], 0],
                 [0, 0, 0, 1]]
        else:
            Z = unik.eye(4, dtype=dtype)
        if s is not None:
            S = [[1, s[0], s[1], 0],
                 [0, 1, s[2], 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
        else:
            S = unik.eye(4, dtype=dtype)

        T = unik.as_tensor(T, dtype=dtype)
        R = unik.as_tensor(R, dtype=dtype)
        Z = unik.as_tensor(Z, dtype=dtype)
        S = unik.as_tensor(S, dtype=dtype)
        return unik.mm(T, unik.mm(R, unik.mm(Z, S)))

    def affine_2d_or_3d(t, r, z, s, dim, dtype):
        return unik.cond(dim == 2,
                         lambda: affine_2d(t, r, z, s, dtype),
                         lambda: affine_3d(t, r, z, s, dtype))

    # Unstack
    prm = unik.flatten(unik.as_tensor(prm))
    dtype = unik.dtype(prm)
    nb_prm = unik.size(prm)
    nb_t = dim
    nb_r = dim*(dim-1) // 2
    nb_z = dim
    nb_s = dim*(dim-1) // 2
    idx = 0
    prm_t = unik.cond(nb_prm > idx, lambda: prm[idx:idx+nb_t], lambda: None)
    idx = idx + nb_t
    prm_r = unik.cond(nb_prm > idx, lambda: prm[idx:idx+nb_r], lambda: None)
    idx = idx + nb_r
    prm_z = unik.cond(nb_prm > idx, lambda: prm[idx:idx+nb_z], lambda: None)
    idx = idx + nb_s
    prm_s = unik.cond(nb_prm > idx, lambda: prm[idx:idx+nb_s], lambda: None)

    # Build affine matrix
    mat = affine_2d_or_3d(prm_t, prm_r, prm_z, prm_s, dim, dtype)

    # Multiply with RAS
    if layout is not None:
        mat = unik.matmul(mat, layout_matrix(layout))

    return mat


