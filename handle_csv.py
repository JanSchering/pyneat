# %%
import matplotlib.pyplot as plt
import pandas as pd
import os

data_path = os.path.join(os.getcwd(), "data")


# %%
data = pd.read_csv(os.path.join(data_path, "reward_max.csv"))
plt.plot(data["Step"], data["Value"])
data = pd.read_csv(os.path.join(data_path, "reward_mean.csv"))
plt.plot(data["Step"], data["Value"])
plt.show()
# %%
