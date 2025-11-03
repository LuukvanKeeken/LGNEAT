import sympy as sp
import numpy as np

SCALE = 3

class SympySigmoid(sp.Function):
    @classmethod
    def eval(cls, z):
        z = 1 / (1 + sp.exp(-z))
        return z


class SympyScaledSigmoid(sp.Function):
    @classmethod
    def eval(cls, z):
        return SympySigmoid(z) * SCALE


class SympyTanh(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.tanh(z)


class SympyScaledTanh(sp.Function):
    @classmethod
    def eval(cls, z):
        return SympyTanh(z) * SCALE


class SympySin(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.sin(z)


class SympyRelu(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Max(z, 0)


class SympyLelu(sp.Function):
    @classmethod
    def eval(cls, z):
        leaky = 0.005
        return sp.Piecewise((z, z > 0), (leaky * z, True))


class SympyIdentity(sp.Function):
    @classmethod
    def eval(cls, z):
        return z


class SympyInv(sp.Function):
    @classmethod
    def eval(cls, z):
        z = sp.Piecewise((sp.Max(z, 1e-7), z > 0), (sp.Min(z, -1e-7), True))
        return 1 / z


class SympyLog(sp.Function):
    @classmethod
    def eval(cls, z):
        z = sp.Max(z, 1e-7)
        return sp.log(z)


class SympyExp(sp.Function):
    @classmethod
    def eval(cls, z):
        z = SympyClip(z, -10, 10)
        return sp.exp(z)


class SympyAbs(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Abs(z)


class SympyClip(sp.Function):
    @classmethod
    def eval(cls, val, min_val, max_val):
        if val.is_Number and min_val.is_Number and max_val.is_Number:
            return sp.Piecewise(
                (min_val, val < min_val), (max_val, val > max_val), (val, True)
            )
        return None

    @staticmethod
    def numerical_eval(val, min_val, max_val, backend=np):
        return backend.clip(val, min_val, max_val)

    def _sympystr(self, printer):
        return f"clip({self.args[0]}, {self.args[1]}, {self.args[2]})"

    def _latex(self, printer):
        return rf"\mathrm{{clip}}\left({sp.latex(self.args[0])}, {self.args[1]}, {self.args[2]}\right)"
    

class SympyGaussian(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.exp(-z**2)


class SympyCos(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.cos(z)
    

class SympySquare(sp.Function):
    @classmethod
    def eval(cls, z):
        return z**2
    

class SympyAbsRoot(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.sqrt(sp.Abs(z))


class SympyNand(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("NAND function requires exactly two inputs.")
        return 1 - z.args[0] * z.args[1]
    

class SympyNor(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("NOR function requires exactly two inputs.")
        return 1 - sp.Max(z.args[0], z.args[1])
    

class SympyAnd(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("AND function requires exactly two inputs.")
        return z.args[0] * z.args[1]
    

class SympyOr(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("OR function requires exactly two inputs.")
        return sp.Max(z.args[0], z.args[1])
    

class SympyFalse(sp.Function):
    @classmethod
    def eval(cls, z):
        return 0
    

class SympyAAndNotB(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("A AND NOT B function requires exactly two inputs.")
        return z.args[0] * (1 - z.args[1])


class SympyAGate(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("A gate function requires exactly two inputs.")
        return z.args[0]
    

class SympyXor(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("XOR function requires exactly two inputs.")
        return z.args[0] + z.args[1] - 2 * z.args[0] * z.args[1]
    

class SympyXnor(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("XNOR function requires exactly two inputs.")
        return 1 - (z.args[0] + z.args[1] - 2 * z.args[0] * z.args[1])
    

class SympyNotAGate(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("NOT A gate function requires exactly two inputs.")
        return 1 - z.args[0]
    

class SympyAOrNotB(sp.Function):
    @classmethod
    def eval(cls, z):
        if len(z.args) != 2:
            raise ValueError("A OR NOT B function requires exactly two inputs.")
        return sp.Max(z.args[0], 1 - z.args[1])
    

class SympyTrue(sp.Function):
    @classmethod
    def eval(cls, z):
        return 1