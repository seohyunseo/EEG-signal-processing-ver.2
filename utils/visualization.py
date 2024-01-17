# Function for adding specific arrows on the plot data
def add_arrows(axes, pln_freqs):
    """Add some arrows at 50 Hz and its harmonics."""
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in pln_freqs:
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4) : (idx + 5)].max()
            ax.arrow(
                x=freqs[idx],
                y=y + 18,
                dx=0,
                dy=-12,
                color="red",
                width=0.1,
                head_width=3,
                length_includes_head=True,
            )

# Function for comparison between the original signal and processed signal
def plot_comparison(raw, target_raw, type=None, fmax=50, process_name=None, arrow_points=None):
    for title, data in zip(["Original", process_name], [raw, target_raw]):
        if type == 'psd':       
            fig = data.compute_psd(fmax=fmax).plot(
                average=True, amplitude=False, picks="data", exclude="bads"
            )
            if arrow_points is not None:
                add_arrows(fig.axes[:2], arrow_points)

        elif type == 'raw':
            fig = data.plot()

        fig.suptitle("{}".format(title), size="x-large", weight="bold")