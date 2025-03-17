# %% [markdown]
# # tutorial for *ATOL: Automatic Topologically-Oriented Learning*

# %% [markdown]
# __Author:__ Martin Royer

# %% [markdown]
# ## Outline:
# In this notebook:
# - select a graph dataset that exists in the /perslay/ submodule
# - generate the associated persistence diagrams
# - show an example of centers and ATOL-features
# - run a ten-fold classification experiment solely based on the resulting ATOL graph features

# %% [markdown]
# ### Select problem and budget

# %%
import os
import numpy as np
from itertools import product

graph_problem = "MUTAG"
graph_folder = "../../perslay/tutorial/data/" + graph_problem + "/" # this should point to a perslay repository

# %% [markdown]
# ### Compute HKS-extended persistence diagrams for this problem

# %%
from atol.utils import compute_tda_for_graphs
print("- [%s] TDA computation" % graph_problem)
filtrations=['0.1-hks', '10.0-hks']
compute_tda_for_graphs(graph_folder=graph_folder, filtrations=filtrations)

# %% [markdown]
# ### Compute centers and features, plot

# %%
from atol.utils import graph_dtypes, csv_toarray, atol_feats_graphs
from atol import Atol
from sklearn.cluster import MiniBatchKMeans

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

num_elements = len(os.listdir(graph_folder + "mat/"))
all_diags = {}  # load all filtrations into memory once for all
for dtype, gid, filt in product(graph_dtypes, np.arange(num_elements), filtrations):
    all_diags[(dtype, filt, gid)] = csv_toarray(graph_folder + "diagrams/%s/graph_%06i_filt_%s.csv" % (dtype, gid, filt))

atol_objs = {(dtype, filt): Atol(quantiser=MiniBatchKMeans(n_clusters=10)) for dtype, filt in product(graph_dtypes, filtrations)}
for dtype, filt in product(graph_dtypes, filtrations) :
    atol_objs[(dtype, filt)].fit([all_diags[(dtype, filt, gid)] for gid in np.arange(num_elements)])

centers_df = []
for dtype, filt in product(graph_dtypes, filtrations):
    clustercenters = atol_objs[(dtype, filt)].centers
    [centers_df.append({"center": _, "x": clustercenters[_, 0], "y": clustercenters[_, 1], "dtype": dtype, "filt": filt}) for _ in range(clustercenters.shape[0])]
centers_df = pd.DataFrame(centers_df)

for filt in filtrations:
    g = sns.relplot(x="x", y="y", hue="center", col="dtype", data=centers_df[centers_df["filt"] == filt])
    g.fig.suptitle("Filtration %s" % filt)

# %%
feats = pd.DataFrame(atol_feats_graphs(graph_folder, all_diags, atol_objs),
                     columns=["index", "type", "center", "value", "label"])

import seaborn as sns
sns.set()

from sklearn.preprocessing import MinMaxScaler
def renormalize(df):
    df["value"] = MinMaxScaler().fit_transform(df["value"].values.reshape(-1, 1))
    return df

sns.relplot(x="index", y="value", kind="line", hue="center", col="type", col_wrap=2, legend="full",
            data=feats.groupby(["type"]).apply(renormalize))

# %% [markdown]
# ### Ten-fold classification evaluation

# %%
from atol.utils import graph_tenfold
print("- [%s] RF classification" % graph_problem)
vfold_scores, feature_times = graph_tenfold(graph_folder, filtrations)
print("- Crossval ended with avg %.4f, sd %.4f" % tuple(f(vfold_scores) for f in [np.mean, np.std]))
print("- Featurisation took %.3f ms" % (np.mean(feature_times) * 1000.0))
print("- [%s] end\n" % graph_problem)

# %%



