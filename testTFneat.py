# %%
from IPython.display import Image
from tfpyneat import TFNode, TFGenome
from tfpyneat import innovationcounter
import tensorflow as tf
import numpy as np
import pickle

innovationcounter.init()
innovationhistory = []
genome = TFGenome(num_in=4, num_out=2, crossover=False)
genome.fully_connect(innovationhistory=innovationhistory)
genome.create_model()

inputs = np.array([1., 2., 3., 4.], dtype="float32")
in2d = inputs[np.newaxis, :]
processed = [in2d[:, col] for col in range(in2d.shape[1])]

print(genome.model.predict(processed))
print(genome.forward(inputs))

# %%
genome.add_node(innovationhistory)
genome.create_model()

inputs = np.array([1., 2., 3., 4.], dtype="float32")
in2d = inputs[np.newaxis, :]
processed = [in2d[:, col] for col in range(in2d.shape[1])]

print(genome.model.predict(processed))
print(genome.forward(inputs))


# %%
# Visualize the base network
tf.keras.utils.plot_model(genome.model, to_file='Model1.png', show_shapes=True)
Image(retina=True, filename='Model1.png')

# %%
# Randomly add nodes and connections
for i in range(20):
    genome.add_node(innovationhistory)

for i in range(10):
    genome.add_connection(innovationhistory)

genome.create_model()
tf.keras.utils.plot_model(genome.model, to_file='Model2.png', show_shapes=True)
Image(retina=True, filename='Model2.png')

inputs = np.array([1., 2., 3., 4.], dtype="float32")
in2d = inputs[np.newaxis, :]
processed = [in2d[:, col] for col in range(in2d.shape[1])]

print(genome.model.predict(processed))
print(genome.forward(inputs))

# %%
# Test pickling process
model = genome.model
with open(f"test.pickle", "wb") as file:
    del genome.model
    pickle.dump(genome, file)

with open(f"test.pickle", "rb") as file:
    loaded_genome = pickle.load(file)

loaded_genome.create_model()
print(model.predict(processed))
print(genome.forward(inputs))
print(loaded_genome.model.predict(processed))
print(loaded_genome.forward(inputs))

# %%
