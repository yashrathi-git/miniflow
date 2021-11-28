from rich.console import Console
from rich.progress import (BarColumn, Progress, ProgressColumn, Task,
                           TimeRemainingColumn)
from rich.text import Text

from miniflow.activations import ActivationSoftmax
from miniflow.layers import LayerDropout, LayerInput
from miniflow.loss import CategoricalLossEntropy, CommonSoftmaxCrossEntropyLoss

console = Console()


class EpochSpeedColumn(ProgressColumn):
    """Renders human readable epoch speed."""

    def render(self, task: Task) -> Text:
        """Show epoch speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        epoch_speed = f"{speed:.2f} Epochs/s"
        return Text(epoch_speed, style="progress.data.speed")


@contextlib.contextmanager
def handle_keyboard_interrupt():
    """Handle keyboard interrupt."""
    try:
        yield
    except KeyboardInterrupt:
        console.print("[red]Exiting...")
        exit(1)


class Model:
    def __init__(self):
        self.layers = []
        # These are to be set by set method, or by attributes
        self.loss = None
        self.optimizer = None
        self.trainable_layers = None
        self.output_activation = None
        self.accuracy = None
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(
        self, X, Y, *, epochs=1, print_every=1, validation_data=None, progress_bar=True
    ):
        progress = Progress(
            "[progress.description]{task.description}",
            "[green]Epoch [purple]{task.completed}/{task.total}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            EpochSpeedColumn(),
            console=console,
        )
        self.accuracy.init(Y)
        # Don't display progress bar if there progress_bar is False
        with ExitStack() as stack:
            if progress_bar:
                progress = stack.enter_context(progress)
            stack.enter_context(handle_keyboard_interrupt())
            for epoch in progress.track(
                range(epochs + 1), total=epochs, description="Training"
            ):
                output = self.forward(X, training=True)
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
                    progress.console.print(
                        f"[green]Epoch[/]: {epoch}/{epochs}, "
                        f"[green]Accuracy[/]: {accuracy:.2f}, "
                        f"[green]Loss[/]: {total_loss:.4f} ("
                        f"[green]Data Loss[/]: {data_loss:.4f}, "
                        f"[green]Reg Loss[/]: {reg_loss:.4f}), "
                        f"[green]Learning Rate[/]: {self.optimizer.current_learning_rate:.4f}",
                        highlight=False,
                    )
        if validation_data:
            self.validation_data(*validation_data)

    def validation_data(self, X, Y):
        output = self.forward(X, training=False)
        loss = self.loss.calculate(output, Y, include_regularization=False)
        prediction = self.output_activation.predictions(output)
        accuracy = self.accuracy.calculate(prediction, Y)
        console.print(f"\n[green]======= Validation Set =======[/]")
        console.print(f"[green]Accuracy[/]: {accuracy:.2f}, [green]Loss[/]: {loss:.4f}")

    def backward(self, output, Y):
        if self.softmax_classifier_output is not None:
            # Speed up the calculation as the formula for grad of combined Softmax and
            # CrossEntropy Loss is much more simple
            self.softmax_classifier_output.backward(output, Y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            layers = self.layers[:-1]
        else:
            layers = self.layers
            self.loss.backward(output, Y)

        for layer in reversed(layers):
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

        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(
            self.loss, CategoricalLossEntropy
        ):
            self.softmax_classifier_output = CommonSoftmaxCrossEntropyLoss()

        self.loss.trainable_layers = self.trainable_layers

    def forward(self, X, training):
        self.inp_layer.forward(X)
        for layer in self.layers:
            params = [layer.prev.output]
            if isinstance(layer, LayerDropout):
                params.append(training)
            layer.forward(*params)
        return layer.output
