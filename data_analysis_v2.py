import mne
import matplotlib.pyplot as plt
from utils.visualization import plot_comparison

ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
            'P1', 'Pz', 'P2', 'POz']

event_dict={
 'reject':1,
 'eye move':2,
 'eye open':3,
 'eye close':4,
 'new run':5,
 'new trial':6,
 'class 1':7,
 'class 2':8,
 'class 3':9,
 'class 4':10,
}

raw = mne.io.read_raw_gdf('./data/A01T.gdf',
                         eog=['EOG-left', 'EOG-central', 'EOG-right'], preload=True)
raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

for i in range(len(ch_names)):
    raw.rename_channels({
        raw.info["ch_names"][i]: ch_names[i]
    })

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Use the average of all channels as reference
ref_avr = "average"

# Conduct CAR (Common Average Reference)
raw_ref = raw.copy().set_eeg_reference(ref_channels=ref_avr)

events = mne.events_from_annotations(raw_ref)

# Extracts epochs of 3s time period from the datset into 288 events for all 4 classes 
tmin = -1.
tmax = 2.
epochs = mne.Epochs(raw_ref, events[0], event_id=[7,8,9,10],tmin= tmin, tmax=tmax, preload=True, baseline=None)

# Baseline correction
baseline = (None, 0)
epochs.apply_baseline(baseline)

# Filter the raw signal with a band pass filter in 0.5 - 50.0 Hz
l_freq = .05
h_freq = 50.0
epochs.filter(l_freq, h_freq, picks='eeg')

# Create ICA object first
ica_obj = mne.preprocessing.ICA(
                    n_components=0.99,
                    method='infomax',
                    max_iter="auto",
                    random_state=1,
                    fit_params=dict(extended=True))

ica_obj.fit(epochs)

