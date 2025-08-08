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

Systems neuroscience aims to understand how information is encoded in populations of neurons.
In modern experiments, this often involves recording large and complex datasets.
There is a growing need for novel statistical methods
to analyze these data.
Open neuroscience datasets are a great tool for developing and testing
such methods. This post series provides quick tutorials on how to access and
use some open datasets for neural population coding using Python.
Specifically, we'll obtain neural population spike counts 
across experimental conditions and trials. This first tutorial
looks at the [Allen Institute - Visual Coding](https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels)
dataset.

## Motivation

There are many open datasets, data formats, and processing tools
for systems neuroscience.
While this diversity is valuable, it can also be overwhelming--especially
when you're trying to find a dataset that fits your needs or make
sense of complex documentation.

This post aims to provide minimal code to obtain
one specific type of neural population data:
**spike counts across trials and conditions**.
Other types of
data that we are not covering here include temporal data
(spike trains) and data without trials (e.g., spontaneous
activity, free behavior, etc.).

The data format that we aim for is an array $$X$$ of shape
$$(t, n)$$, where $$t$$ is the number of trials and $$n$$ is the number
of neurons, and a vector $$y$$ of length $$t$$, indicating
the experimental condition for each trial. Each entry in
$$X$$ contains the number of spikes of a neuron in a trial.
We will also show
a very simple analysis on the datasets, applying LDA
for decoding and visualization.


## Allen Institute - Brain Observatory

The Allen Institute provides many open datasets. One that fits our
goal is [Visual Coding - Neuropixels](https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels).
This dataset contains recordings from the mouse brain
during visual stimulation experiments, using the Neuropixels probe
(one of the latest technologies for large neural recordings).

As can be seen in the [dataset documentation](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html),
the data is available as
[Neurodata Without Borders](https://nwb.org/) (NWB) files, which
are a standard format for storing neural data (we'll see more about NWB
below). However, there is also a Python package called
[AllenSDK](https://allensdk.readthedocs.io/en/latest/) that
provides a convenient interface for the data. 

First, let's install the package using `pip` in the
command line (we recommend using a virtual environment):

```bash
pip install allensdk
```

Next, we use the package to access the dataset.

### The AllenSDK cache and downloading single session data

Importantly, the full dataset is very large (855 GB), so
`allensdk` provides a convenient way to download only the data
that we need (e.g. only one of the 58 sessions). The
tool for this is an object called `EcephysProjectCache`,
which we can use to manage and download the dataset. 

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

The variable `my_ses` is an object of type `EcephysSession`,
which allows us to conveniently access the data
(see [a tutorial from AllenSDK](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_session.html)).

### Filtering trials by stimulus properties

What do we mean by "conveniently access the data"?
A given experimental session has a lot of data, of which we
might only want a small part. For example, many different
types of stimuli are presented in a given session, like
static gratings, gabors, natural images and natural movies,
as shown in the diagram below (obtained from the Allen Brain
Observatory [cheat sheet](https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/0f/5d/0f5d22c9-f8f6-428c-9f7a-2983631e72b4/neuropixels_cheat_sheet_nov_2019.pdf)).

![](/files/blog/neural_data/allen_drawing.png)


In this tutorial, we'll focus on responses to static gratings--these
come in 6 orientations, 5 spatial frequencies, and 4 phases,
totaling 120 unique conditions.
For this, we use the attribute `my_ses.stimulus_presentations`, which has a
Pandas DataFrame with the stimulus information for each trial.
Let's visualize the first few rows of this DataFrame:

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

To extract the neural data for the trials that we want, we
can use the trial index in `stim_table` (the value under the header
`stimulus_presentation_id`). For that, let's filter `stim_table` to
only have the trials with the static gratings stimulus using
`stimulus_name`.
We also exclude trials with a null stimulus (i.e., where no stimulus was presented).

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
one phase (0 degrees).


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

### Filtering neurons by brain area and firing rate

The Visual Coding - Neuropixels dataset contains
recordings from many brain areas and neurons. 
For a given analyses, we might want to focus on neurons from
a specific brain area, that satisfy some quality metrics.
(See this [quality metrics tutorial](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html)
from AllenSDK).

Like we did with stimulus presentations, we can use the
session object to filter the neurons that we want. We
start by obtaining the table with the neuron information:

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

Let's get the indices of only the units in the primary visual
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


### Obtaining the spike counts

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

Given that we only kept one spatial frequency and
phase, the experimental condition for each trial
is determined by the orientation. We can extract
the orientations from `stim_table` and convert them
to integer labels:

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


### Simple decoding and visualization analysis

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

