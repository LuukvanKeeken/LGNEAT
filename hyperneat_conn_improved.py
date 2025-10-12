from tensorneat.algorithm.hyperneat.hyperneat import HyperNEATConn


class HyperNEATConnImproved(HyperNEATConn):

    custom_attrs = ["weight"]

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        in_idx, out_idx, weight = conn

        in_idx = int(in_idx)
        out_idx = int(out_idx)
        weight = round(float(weight), precision)

        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}}, weight: {:<{float_width}})".format(
            self.__class__.__name__,
            in_idx,
            out_idx,
            weight,
            idx_width=idx_width,
            float_width=precision + 3,
        )