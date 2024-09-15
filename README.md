# Miniflow

## Install Miniflow
```
pip install git+https://github.com/yashrathi-git/miniflow
```

## Example Usage

### Create a basic model and train it
```py
from miniflow.core import Model
from miniflow.layers import LayerDense
from miniflow.activations import ActivationReLU, ActivationSoftmax
from miniflow.loss import CategoricalLossEntropy
from miniflow.optimizers import AdamOptimizer
from miniflow.accuracy import AccuracyCategorical

# Create model
model = Model()

# Add layers
model.add(LayerDense(input_features, 64))
model.add(ActivationReLU())
model.add(LayerDense(64, 32))
model.add(ActivationReLU())
model.add(LayerDense(32, output_classes))
model.add(ActivationSoftmax())

# Set loss, optimizer, and accuracy
model.set(
    loss=CategoricalLossEntropy(),
    optimizer=AdamOptimizer(learning_rate=0.001),
    accuracy=AccuracyCategorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X_train, y_train, epochs=1000, validation_data=(X_val, y_val))
```

Deep learning framework from scratch for vanilla neural networks.
Made while following along [coursera's DL specialisation](https://www.coursera.org/learn/deep-neural-network/home/welcome) 
with help of [nnfs source code](https://github.com/Sentdex/nnfs_book/)
