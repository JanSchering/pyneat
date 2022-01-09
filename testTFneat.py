# %%
from IPython.display import Image
from tfpyneat import TFNode, TFGenome
from pyneat import innovationcounter
import tensorflow as tf
import numpy as np

innovationcounter.init()
innovationhistory = []
genome = TFGenome(num_in=4, num_out=2, crossover=False)
genome.fully_connect(innovationhistory=innovationhistory)
genome.create_model()

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

# %%
# Try forwarding an input through the new network
output = genome.forward(np.array([1, 2, 3, 4])[np.newaxis, :])
# Comes in shape [
#   [sample1_out1, sample1_out2],
#   [sample2_out1, sample2_out2]
# ]
print(output)

# %%
