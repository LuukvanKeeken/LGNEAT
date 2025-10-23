import numpy as np
import sympy as sp


class SympySum(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Add(*z)


class SympyProduct(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Mul(*z)


class SympyMax(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Max(*z)


class SympyMin(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Min(*z)


class SympyMaxabs(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Max(*z, key=sp.Abs)


class SympyMean(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Add(*z) / len(z)
    

class FilterNans(sp.Function):
    @classmethod
    def eval(cls, z):
        # Assuming z is a list of sympy expressions, filter out NaNs
        filtered = [zi for zi in z if not zi.has(sp.nan)]
        if len(filtered) == 2:
            return sp.Matrix(filtered)
        else:
            raise ValueError("FilterNans expects exactly two non-NaN values.")


class SympyArgmax(sp.Function):
    @classmethod
    def eval(cls, z):
        if not z:
            raise ValueError("Argmax requires at least one input.")
        max_index = 0
        max_value = z[0]
        for i in range(1, len(z)):
            if z[i] > max_value:
                max_value = z[i]
                max_index = i
        return sp.Integer(max_index)