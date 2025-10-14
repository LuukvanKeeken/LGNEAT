from tensorneat.src.tensorneat.algorithm.hyperneat.hyperneat import HyperNEATConn


class HyperNEATConnImprovedLEO(HyperNEATConn):

    custom_attrs = ["weight"]

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        in_idx, out_idx, weight, leo_value = conn

        in_idx = int(in_idx)
        out_idx = int(out_idx)
        weight = round(float(weight), precision)
        leo_value = round(float(leo_value), precision)

        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}}, weight: {:<{float_width}}, leo: {:<{float_width}})".format(
            self.__class__.__name__,
            in_idx,
            out_idx,
            weight,
            leo_value,
            idx_width=idx_width,
            float_width=precision + 3,
        )
    
    def to_dict(self, state, conn):
        in_idx, out_idx, weight, leo_value = conn[:4]
        return {
            "in": int(in_idx),
            "out": int(out_idx),
            "weight": float(weight),
            "leo": float(leo_value),
        }