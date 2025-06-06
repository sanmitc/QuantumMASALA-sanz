from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
__all__ = [
    "WavefunGType",
    "get_WavefunG",
    "WavefunRType",
    "get_WavefunR",
    "WavefunType",
]

from functools import lru_cache
from typing import Union
import numpy as np

from qtm.gspace import GkSpace
from .buffer import BufferType
from .field import get_FieldR
from .gemm_wrappers import get_zgemm
from qtm.config import MPI4PY_INSTALLED

from qtm.config import NDArray
from qtm.msg_format import *


class WavefunGType(BufferType):
    gspc: GkSpace = None
    gkspc: GkSpace = None
    numspin: int = None
    basis_type = "g"
    basis_size: int = None
    ndarray: type = None

    def __init_subclass__(cls, gkspc: GkSpace, numspin: int):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(type_mismatch_msg("gkspc", gkspc, GkSpace))
        if not isinstance(numspin, int) or numspin <= 0:
            raise TypeError(type_mismatch_msg("numspin", numspin, "a positive integer"))
        cls.gspc, cls.gkspc = gkspc, gkspc
        cls.numspin, cls.basis_size = numspin, numspin * gkspc.size_g
        cls.ndarray = type(cls.gspc.g_cryst)

    @classmethod
    def zgemm(
        cls,
        a: NDArray,
        b: NDArray,
        trans_a: Literal[0, 1, 2] = 0,
        trans_b: Literal[0, 1, 2] = 0,
        alpha: complex = 1.0,
        out: NDArray | None = None,
        beta: complex = 0.0,
    ) -> NDArray:
        """Implements C := alpha*op( A )*op( B ) + beta*C, where op( X ) is
        one of: op( X ) = X or op( X ) = X**T or op( X ) = X**H

        Array arguments are expected to be in Fortran contiguous order.
        `trans_a` and `trans_b` take 0, 1 or 2 where:

        * 0 -> op( X ) = X
        * 1 -> op( X ) = X**T (transpose)
        * 2 -> op( X ) = X**H (transpose conjugate)

        Parameters
        ----------
        a : NDArray
        b : NDArray
        trans_a: Literal[0, 1, 2], default=0
        trans_b: Literal[0, 1, 2], default=0
        alpha: complex, default=1.0
        out: Optional[NDArray], default=None
            If None, a new array is allocated and C above is taken to be zero
        beta: complex, default=0.0

        Returns
        -------
        out : NDarray
        """
        return get_zgemm(cls.ndarray)(a, b, trans_a, trans_b, alpha, out, beta)

    def to_r(self) -> WavefunRType:
        wfn_r = get_WavefunR(self.gkspc, self.numspin).empty(self.shape)
        self.gkspc._g2r(self._data, wfn_r._data)
        return wfn_r

    def to_g(self) -> WavefunGType:
        return self

    def norm2(self) -> NDArray:
        """Returns the norm-squared value of the wavefunction vectors."""
        vdot = self.gspc.allocate_array(self.shape)
        for iwfn, wfn in enumerate(self.data.reshape((-1, self.basis_size))):
            vdot.ravel()[iwfn] = np.vdot(wfn, wfn)

        if hasattr(self.gkspc, "pwgrp_comm"):
            comm = self.gkspc.pwgrp_comm
            comm.Allreduce(comm.IN_PLACE, vdot, comm.SUM)
        return vdot

    def norm(self) -> NDArray:
        """Returns the norm of the wavefunction vectors."""
        return np.sqrt(self.norm2())

    def normalize(self):
        """Normalizes the wavefunction so that its norm is 1."""
        self.data[:] /= self.norm()[:, np.newaxis]

    def vdot(self, ket: get_WavefunG, slice_partial_basis: slice = None) -> NDArray:
        """Evaluates the vector dot product between the wavefunctions in two
        `WavefunG` instances. The values of the instance are conjugated, not
        the ones in the method input.

        Parameters
        ----------
        ket : WavefunG
            Wavefunction to evaluate the dot product with

        Returns
        -------
        NDArray
            Array containing the vector dot product between the wavefunctions
            in instance and the ones in the input argument.
        """
        # TODO: Reprofile this and optimize it for memory usage
        # TODO: Reevaluate this implementation when SciPy is compiled with CBLAS too
        # TODO: Add check for C-Contiguity
        if not isinstance(ket, type(self)):
            raise TypeError(type_mismatch_msg("ket", ket, type(self)))
        if self.gkspc is not ket.gkspc:
            raise ValueError(
                obj_mismatch_msg("self.gkspc", self.gkspc, "ket.gkspc", ket.gkspc)
            )
        if slice_partial_basis is None:
            bra_ = self.data.reshape((-1, self.basis_size))
            ket_ = ket.data.reshape((-1, self.basis_size))
            braket = self.zgemm(
                alpha=1.0,
                a=bra_.T,
                trans_a=2,
                b=ket_.T,
                trans_b=0,
            )
            # braket = bra_.conj() @ ket_.T

            if self.shape == ():
                return braket.reshape(ket.shape)
            elif ket.shape == ():
                return braket.reshape(self.shape)
            else:
                return braket.reshape((*self.shape, *ket.shape))
        else:
            bra_ = self.data.reshape((-1, self.basis_size))[:, slice_partial_basis]
            ket_ = ket.data.reshape((-1, ket.basis_size))[:, slice_partial_basis]
            braket = self.zgemm(
                alpha=1.0,
                a=bra_.T,
                trans_a=2,
                b=ket_.T,
                trans_b=0,
            )
            if self.shape == ():
                return braket.reshape(ket.shape)
            elif ket.shape == ():
                return braket.reshape(self.shape)
            else:
                return braket.reshape((*self.shape, *ket.shape))


class WavefunRType(BufferType):
    gspc: GkSpace = None
    gkspc: GkSpace = None
    numspin: int = None
    basis_type = "r"
    basis_size: int = None
    ndarray: type = None
    zgemm = None

    def __init_subclass__(cls, gkspc: GkSpace, numspin: int):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(type_mismatch_msg("gkspc", gkspc, GkSpace))
        if not isinstance(numspin, int) or numspin <= 0:
            raise TypeError(type_mismatch_msg("numspin", numspin, "a positive integer"))
        cls.gspc, cls.gkspc = gkspc, gkspc
        cls.numspin, cls.basis_size = numspin, numspin * gkspc.size_r
        cls.ndarray = type(cls.gspc.g_cryst)
        cls.zgemm = get_zgemm(cls.ndarray)

    def to_g(self) -> WavefunGType:
        wfn_g = get_WavefunG(self.gkspc, self.numspin).empty(self.shape)
        self.gkspc._r2g(self._data, wfn_g._data)
        return wfn_g

    def to_r(self) -> WavefunRType:
        return self

    def get_density(self, normalize: bool = True) -> get_FieldR:
        """ "Constructs the probability density from the stored wavefunctions

        Parameters
        ----------
        normalize : bool, default=True
            If True, normalize the densities so that they integrate to one.

        Returns
        -------
        FieldR
            Represents the probability density of the wavefunctions.
        """
        gwfn = self.gkspc.gwfn
        den_data = self.data.conj()
        den_data *= self.data
        den_data = den_data.reshape((*self.shape, self.numspin, -1))
        den = get_FieldR(gwfn)(den_data)
        if not normalize:
            return den
        den /= den.integrate_unitcell()[:, np.newaxis]
        return den


WavefunType = Union[WavefunGType, WavefunRType]


@lru_cache(maxsize=None)
def get_WavefunG(gkspc: GkSpace, numspin: int) -> type[WavefunGType]:
    if MPI4PY_INSTALLED:
        from qtm.mpi.gspace import DistGkSpace
        from qtm.mpi.containers import get_DistWavefunG

        if isinstance(gkspc, DistGkSpace):
            return get_DistWavefunG(gkspc, numspin)

    class WavefunG(WavefunGType, gkspc=gkspc, numspin=numspin):
        pass

    return WavefunG


@lru_cache(maxsize=None)
def get_WavefunR(gkspc: GkSpace, numspin: int) -> type[WavefunRType]:
    if MPI4PY_INSTALLED:
        from qtm.mpi.gspace import DistGkSpace
        from qtm.mpi.containers import get_DistWavefunR

        if isinstance(gkspc, DistGkSpace):
            return get_DistWavefunR(gkspc, numspin)

    class WavefunR(WavefunRType, gkspc=gkspc, numspin=numspin):
        pass

    return WavefunR
