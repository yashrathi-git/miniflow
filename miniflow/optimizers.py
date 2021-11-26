import numpy as np

from miniflow.layers import LayerDense


class BaseOptimizer:
    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 1

    def pre_update_model(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def post_update_model(self):
        self.iterations += 1

    def _update_params(self, layer, dw, db):
        layer.weights = layer.weights - self.current_learning_rate * dw
        layer.biases = layer.biases - self.current_learning_rate * db


class SDGOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        super(SDGOptimizer, self).__init__(learning_rate, decay)
        self.momentum = momentum

    def update_params(self, layer: LayerDense):
        if self.momentum:
            if not hasattr(layer, "rolling_dw"):
                layer.rolling_dw = np.zeros_like(layer.weights)
                layer.rolling_db = np.zeros_like(layer.biases)
            dw = self.momentum * layer.rolling_dw + (1 - self.momentum) * layer.dweights
            db = self.momentum * layer.rolling_db + (1 - self.momentum) * layer.dbias
            layer.rolling_db = db
            layer.rolling_dw = dw

            # Bias correction
            dw = dw / (1 - self.momentum ** self.iterations)
            db = db / (1 - self.momentum ** self.iterations)

        else:
            dw = layer.dweights
            db = layer.dbias
        self._update_params(layer, dw, db)


class AdamOptimizer(BaseOptimizer):
    def __init__(
        self, learning_rate=0.001, decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        super(AdamOptimizer, self).__init__(learning_rate, decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update_params(self, layer: LayerDense):
        if not hasattr(layer, "cache_Sdw"):
            # S represents the RMS values
            # V represents the momentum(rolling avg)
            layer.cache_Sdw = np.zeros_like(layer.weights)
            layer.cache_Sdb = np.zeros_like(layer.biases)
            layer.cache_Vdw = np.zeros_like(layer.weights)
            layer.cache_Vdb = np.zeros_like(layer.biases)
        Vdw = self.beta1 * layer.cache_Vdw + (1 - self.beta1) * layer.dweights
        Vdb = self.beta1 * layer.cache_Vdb + (1 - self.beta1) * layer.dbias
        Sdw = (self.beta2 * layer.cache_Sdw) + (1 - self.beta2) * (layer.dweights ** 2)
        Sdb = (self.beta2 * layer.cache_Sdb) + (1 - self.beta2) * (layer.dbias ** 2)

        layer.cache_Sdw = Sdw
        layer.cache_Sdb = Sdb
        layer.cache_Vdw = Vdw
        layer.cache_Vdb = Vdb

        # Bias correction
        Sdw_corr = Sdw / (1 - self.beta2 ** self.iterations)
        Sdb_corr = Sdb / (1 - self.beta2 ** self.iterations)
        Vdw_corr = Vdw / (1 - self.beta1 ** self.iterations)
        Vdb_corr = Vdb / (1 - self.beta1 ** self.iterations)

        self._update_params(
            layer,
            dw=(Vdw_corr / (np.sqrt(Sdw_corr) + self.epsilon)),
            db=(Vdb_corr / (np.sqrt(Sdb_corr) + self.epsilon)),
        )
