from miniflow.layers import LayerInput


class Model:
    def __init__(self):
        self.layers = []
        # These are to be set by set method, or by attributes
        self.loss = None
        self.optimizer = None
        self.trainable_layers = None
        self.output_activation = None
        self.accuracy = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, Y, *, epochs=1, print_every=1):
        self.accuracy.init(Y)
        for epoch in range(epochs + 1):
            output = self.forward(X)
            data_loss, reg_loss = self.loss.calculate(output, Y)
            total_loss = data_loss + reg_loss
            prediction = self.output_activation.predictions(output)
            accuracy = self.accuracy.calculate(prediction, Y)
            self.backward(output, Y)

            # Optimize
            self.optimizer.pre_update_model()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_model()

            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}/{epochs}, "
                    f"Accuracy: {accuracy:.2f}, "
                    f"Loss: {total_loss:.4f}("
                    f"Data Loss: {data_loss:.4f}, "
                    f"Reg Loss: {reg_loss:.4f}), "
                    f"Learning Rate: {self.optimizer.current_learning_rate:.4f}"
                )

    def backward(self, output, Y):
        self.loss.backward(output, Y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def finalize(self):
        self.inp_layer = LayerInput()
        self.trainable_layers = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.prev = self.inp_layer
                layer.next = self.layers[i + 1]
            elif i < len(self.layers) - 1:
                layer.prev = self.layers[i - 1]
                layer.next = self.layers[i + 1]
            else:
                layer.prev = self.layers[i - 1]
                layer.next = self.loss
                self.output_activation = layer
            if hasattr(layer, "weights"):
                self.trainable_layers.append(layer)

        self.loss.trainable_layers = self.trainable_layers

    def forward(self, X):
        self.inp_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.prev.output)
        return layer.output
