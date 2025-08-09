---
title: 'Neural Coding Open Datasets: Python Tutorial Part 1'
date: 2025-08-10
permalink: /posts/2025/08/open-neural-coding-datasets/
tags:
  - Neuroscience
  - Open Data
  - Python
  - Neural Coding
---

Understanding how populations of neurons encode information is a
central goal of systems neuroscience.
As neural recordings scale in magnitude, 
there is a growing need for novel statistical methods
to analyze these data.
Open neuroscience datasets are a great tool for developing and testing
such methods. This post kicks off a short series on
working with some open neural population datasets using Python.
We'll focus on one core task: **obtaining spike counts across trials and conditions**.
This first post uses the
[Allen Institute - Visual Coding](https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels)
dataset, accessed via the AllenSDK package.

We'll:

* Explore the dataset using AllenSDK
* Download data for one session
* Filter the data by stimulus presented and neuron properties
* Extract a spike count matrix and condition labels
* Visualize and decode conditions using LDA

We'll also run a quick supervised dimensionality reduction
and decoding pipeline using Linear Discriminant Analysis (LDA).

## Motivation

There are many open datasets and data tools
for systems neuroscience.
While valuable, it can also be overwhelming to navigate the
options and their complex documentations.

This post aims to provide an example of relatively simple
code to achieve a specific task:
**obtain spike counts across trials and conditions**.
Our end goal is to obtain:
* A matrix `X` of shape `(trials, neurons)` with spike counts
* A vector `y` of length `trials` with condition labels

This post will not explain the details of standard
Python packages such as `pandas`, nor the intricacies of
the AllenSDK package or the
[Neurodata Without Borders](https://nwb.org/) (NWB) format.
Instead, it will focus on the steps to obtain the spike counts
and condition labels.


## Allen Institute - Brain Observatory

We'll use the Allen Institute 
[Visual Coding - Neuropixels](https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels)
dataset. This dataset
contains high-density recordings from the mouse brain
during visual stimulation experiments, using the Neuropixels probe
(one of the latest technologies for large neural recordings).

This is a large dataset with many sessions and 855 GB of
NWB files. To interface conveniently with the data, we will use
the Python package [AllenSDK](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html),
developed by the Allen Institute.

First, let's install the package using `pip` in the
command line (we recommend using a virtual environment):

```bash
pip install allensdk
```

Next, we use the package to access the dataset.

## The AllenSDK cache and downloading single session data

The `allensdk` package provides a cache system to manage the
data. For example, it allows us to obtain the metadata for
the sessions before downloading them.

The tool for this is an object called `EcephysProjectCache`.
Let's create the cache object and use it to download the
metadata for the sessions:

```python
import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

output_dir = './allen_data/allen_cache_dir'  # Where the data will be stored locally
manifest_path = os.path.join(output_dir, "manifest.json")

# Create the cache object, that will manage the data

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# Use the cache to get the session table
# This will create the output_dir above if it doesn't exist,
# and save the session metadata there.

sessions = cache.get_session_table()
```

The new `sessions` variable is a Pandas DataFrame
with information about each session, like the mouse age,
the experimental protocol, and the session ID. Let's
print the first few rows to see what we have:

```python
print(sessions.head())
print(f"Column names: {sessions.columns}")
```

```
                   published_at  specimen_id  ... probe_count                         ecephys_structure_acronyms
id                                            ...                                                               
715093703  2019-10-03T00:00:00Z    699733581  ...           6  [CA1, VISrl, nan, PO, LP, LGd, CA3, DG, VISl, ...
719161530  2019-10-03T00:00:00Z    703279284  ...           6  [TH, Eth, APN, POL, LP, DG, CA1, VISpm, nan, N...
721123822  2019-10-03T00:00:00Z    707296982  ...           6  [MB, SCig, PPT, NOT, DG, CA1, VISam, nan, LP, ...
732592105  2019-10-03T00:00:00Z    717038288  ...           5       [grey, VISpm, nan, VISp, VISl, VISal, VISrl]
737581020  2019-10-03T00:00:00Z    718643567  ...           6      [grey, VISmma, nan, VISpm, VISp, VISl, VISrl]

Column names: Index(['published_at', 'specimen_id', 'session_type', 'age_in_days', 'sex',
       'full_genotype', 'unit_count', 'channel_count', 'probe_count',
       'ecephys_structure_acronyms'],
      dtype='object')
```

Now, let's use the cache to download the actual neural recordings
data for a specific session. We filter the sessions to the ones
with the `brain_observatory_1.1` protocol, and
obtain the ID of the session in the 21st row.

```python
sessions = sessions[(sessions.session_type=='brain_observatory_1.1')]  # Filter sessions
ind = sessions.index.values[21]  # Get the session ID
my_ses = cache.get_session_data(ind)  # Download the data for that session
```

The object `my_ses` is of the class `EcephysSession`,
and it allows us to conveniently access the data, as shown
below (see also [this tutorial from AllenSDK](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_session.html)).

## Filtering trials by stimulus properties

Each session contains many types of stimuli, like
static gratings, gabors, natural images and natural movies,
as shown in the diagram below (obtained from
[this cheat sheet](https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/0f/5d/0f5d22c9-f8f6-428c-9f7a-2983631e72b4/neuropixels_cheat_sheet_nov_2019.pdf)).

![](/files/blog/neural_data/allen_drawing.png)

Here, we'll focus on responses to **static gratings**, which vary in:
* Orientation (0, 30, 60, 90, 120, and 150 degrees)
* Spatial frequency (0.02, 0.04, 0.08, 0.16 and 0.32 cycles/degree)
* Phase (0, 0.25, 0.5 and 0.75 periods)

To select the trials with static gratings, we use
the stimulus information table (a Pandas DataFrame), available as
`my_ses.stimulus_presentations`, which we assign to the variable `stim_table`:

```python
stim_table = my_ses.stimulus_presentations
print(stim_table.head())
print(f"Column names: {stim_table.columns}")
```

```
                         stimulus_block  start_time  stop_time  ...          size   duration stimulus_condition_id
stimulus_presentation_id                                        ...                                               
0                                  null   24.752216  84.818986  ...          null  60.066770                     0
1                                   0.0   84.818986  85.052505  ...  [20.0, 20.0]   0.233520                     1
2                                   0.0   85.052505  85.302704  ...  [20.0, 20.0]   0.250199                     2
3                                   0.0   85.302704  85.552904  ...  [20.0, 20.0]   0.250199                     3
4                                   0.0   85.552904  85.803103  ...  [20.0, 20.0]   0.250199                     4

[5 rows x 16 columns]

Column names: Index(['stimulus_block', 'start_time', 'stop_time', 'temporal_frequency',
       'spatial_frequency', 'contrast', 'phase', 'stimulus_name', 'x_position',
       'frame', 'color', 'y_position', 'orientation', 'size', 'duration',
       'stimulus_condition_id'],
      dtype='object')
```

To obtain the desired trials, we need to extract their trial
index from `stim_table`. For that, let's use
Pandas to filter the DataFrame by `stimulus_name`, and also
exclude trials with a null stimulus (i.e., where no stimulus was presented).

```python
print(f"Number of trials before filtering: {len(stim_table)}")

stim_table = stim_table[
  (stim_table.stimulus_name == "static_gratings") & \
  (stim_table.orientation != "null")
]

print(f"Number of trials after filtering: {len(stim_table)}")
```
```
Number of trials before filtering: 70390
Number of trials after filtering: 5811
```

Let's further simplify our dataset by keeping only
one spatial frequency (0.08 cycles/degree) and
one phase (0 degrees). (This is an arbitrary choice,
you might want to keep more conditions for your analyses.)

```python
stim_table = stim_table[
  (stim_table.spatial_frequency == 0.08) & \
  (stim_table.phase != 0)
]

print(f"Final number of trials: {len(stim_table)}")
```
```
Number of trials after filtering: 1163
```

The `stim_table` DataFrame now has only the trials that we want.
Let's extract the trial indices that we will use shortly to
obtain the neural data:

```python
# Array with indices of desired trials
trial_inds = stim_table.index.values
```

## Filtering neurons by brain area and firing rate

Also, each session contains hundreds of neurons recorded across
many brain areas. Since we are interested in visual coding, let's
just keep the neurons from the primary visual cortex ("VISp").
We'll also focus on neurons with a firing rate above a certain threshold.
(See this [quality metrics tutorial](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html)
from AllenSDK).

Like for the stimulus information, the neurons information is
available as a Pandas DataFrame in `my_ses.units`. We assign
this DataFrame to the variable `units_table` for convenience.

```python
units_table = my_ses.units

print(units_table.head())
print(f"Column names: {units_table.columns}")
print(f"Number of neurons: {len(units_table)}")
```

```
           waveform_PT_ratio  waveform_amplitude  ...  probe_lfp_sampling_rate  probe_has_lfp_data
unit_id                                           ...                                             
951853372           0.510700          232.788465  ...              1249.998592                True
951853379           2.929978           82.579965  ...              1249.998592                True
951853388           0.410656           96.195255  ...              1249.998592                True
951853498           0.434301          103.250355  ...              1249.998592                True
951853596           0.323884           70.187910  ...              1249.998592                True

[5 rows x 40 columns]
Column names: Index(['waveform_PT_ratio', 'waveform_amplitude', 'amplitude_cutoff',
       'cluster_id', 'cumulative_drift', 'd_prime', 'firing_rate',
       'isi_violations', 'isolation_distance', 'L_ratio', 'local_index',
       'max_drift', 'nn_hit_rate', 'nn_miss_rate', 'peak_channel_id',
       'presence_ratio', 'waveform_recovery_slope',
       'waveform_repolarization_slope', 'silhouette_score', 'snr',
       'waveform_spread', 'waveform_velocity_above', 'waveform_velocity_below',
       'waveform_duration', 'filtering', 'probe_channel_number',
       'probe_horizontal_position', 'probe_id', 'probe_vertical_position',
       'structure_acronym', 'ecephys_structure_id',
       'ecephys_structure_acronym', 'anterior_posterior_ccf_coordinate',
       'dorsal_ventral_ccf_coordinate', 'left_right_ccf_coordinate',
       'probe_description', 'location', 'probe_sampling_rate',
       'probe_lfp_sampling_rate', 'probe_has_lfp_data'],
      dtype='object')
Number of neurons: 501
```

Let's see what brain areas are present in the dataset.

```python
print("Brain areas in the dataset:")
print(units_table.ecephys_structure_acronym.unique())
```

```
Brain areas in the dataset:
['DG' 'CA1' 'VISam' 'LP' 'VISpm' 'LGd' 'VISp' 'CA3' 'CA2' 'VISl' 'MB' 'TH'
 'PP' 'PIL' 'VISal']
```

Let's get the indices of the units that are in the primary visual
cortex (VISp), and that have a firing rate above 3 Hz.

```python
min_fr = 3.0  # Minimum firing rate in Hz
units_table = units_table[
  (np.isin(units_table.structure_acronym, "VISp") * \
  units_table.firing_rate > min_fr)
]
v1_inds = units_table.index.values
print(f"Number of neurons in V1 with firing rate > {min_fr}: {len(v1_inds)}")
```

```
Number of neurons in V1 with firing rate > 3.0: 45
```


## Obtaining the spike counts

We have the IDs of the trials and neurons that we want.
The next step is to pass them to the method
`my_ses.presentationwise_spike_counts()`, to obtain
the population responses. We can also pass a
`bin_edges` argument to specify the time window
for counting spikes. Here we use a window from 0.01
to 0.25 seconds after stimulus onset.

```python
X = my_ses.presentationwise_spike_counts(
  bin_edges=(0.01, 0.25),
  stimulus_presentation_ids=trial_inds,
  unit_ids=v1_inds
).values.squeeze()

print(X)
print(f"Shape of spike counts array: {X.shape}")
```

```
[[2 0 2 ... 0 5 4]
 [6 1 1 ... 0 0 6]
 [5 1 4 ... 0 1 7]
 ...
 [5 5 4 ... 0 2 5]
 [0 2 1 ... 2 1 3]
 [0 4 0 ... 0 5 9]]

Shape of spike counts array: (1163, 45)
```

The variable `X` now contains the spike counts for
our 45 neurons across the 1163 selected trials.

Now we should obtain the labels for each trial.
Since we fixed the spatial frequency and phase,
we'll use the orientations that we can extract
from `stim_table`, and convert them to integer labels
using `np.unique()`:

```python
ori = stim_table.orientation.values
# Convert orientations to integer labels
values, y = np.unique(ori, return_inverse=True)

print(f"Orientations: {values}")
print(f"Shape of labels array: {y.shape}")
```

```
Orientations: [0.0 30.0 60.0 90.0 120.0 150.0]
Shape of labels array: (1163,)
```

Now the labels are in `y`, which has the same length
as the number of trials in `X`.


## Simple decoding and visualization analysis

Let's apply a simple analysis to our data
using Linear Discriminant Analysis (LDA).

LDA is a standard technique that can be used for both decoding and
for visualization of data with multiple classes (see my
[previous post about LDA](https://dherrera1911.github.io/posts/2025/01/lda-what-you-should-know/)).

Specifically, we'll use LDA to find
the two dimensions that best separate the different gratings
in neural response space. We'll see what performance we
get for decoding orientations from these features, and visualize
the neural responses in the new feature space.

We'll assume that the `sklearn` package is installed.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt

# Fit LDA with 2 components
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.2, n_components=2)

# Define color map
X_lda = lda.fit_transform(X, y)
colors = plt.cm.tab10.colors  # up to 10 distinguishable colors

plt.figure(figsize=(8, 6))
for i, label in enumerate(values):
    plt.scatter(
        X_lda[y == i, 0],
        X_lda[y == i, 1],
        color=colors[i % len(colors)],
        label=f"{label:.0f}°",
        alpha=0.6,
        edgecolor="k",
        s=40
    )

plt.xlabel("LDA 1")
plt.ylabel("LDA 2")
plt.title("Neural responses projected into LDA space (2D)")
plt.legend(title="Orientation", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()
```

![](/files/blog/neural_data/lda_neuro.png)

In the plot above, the two axes are two dimensions of neural
response space. We can also consider them population activity modes.
Each point corresponds to one trial, and the color indicates
the orientation of the grating. We see that the classes are pretty
well separated, even in this 2D space. Let's evaluate how well we can
decode orientation using the LDA classifier, which amounts to
a linear classifier in the space above:


```python
# Evaluate decoding performance using cross-validation on 2D projections
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.2, n_components=2)
scores = cross_val_score(lda, X, y, cv=5)
print(f"Mean LDA accuracy (on 2D projection, 5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")
```
```
Mean LDA accuracy (on 2D projection, 5-fold CV): 0.931 ± 0.029
```

The result above shows that we can decode the orientation of
the grating to a high accuracy using only the first two LDA components.


## Conclusion

In this post we showed how to access the Allen Institute
Visual Coding - Neuropixels dataset using the `allensdk` package.
We obtained the spike count responses of a population of neurons
to static gratings. We used this dataset to do supervised
dimensionality reduction and neural decoding using LDA.

Future posts will cover other datasets and data formats.

