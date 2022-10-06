def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def plot(ax, data, label=None, color=None, marker=None, markers_on=None):
    mean, ste, runs = data
    base, = ax.plot(mean, label=label, color=color, linewidth=0.8, marker=marker, markevery=markers_on)
    (low_ci, high_ci) = confidenceInterval(mean, ste)
    # ----------- fill-between figures!!! -------------#
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4)

def plot1(ax, data, label=None, color=None):
    mean, ste, runs = data
    base, = ax.plot(mean, label=label, color=color, linewidth=1)
    # (low_ci, high_ci) = confidenceInterval(mean, ste)
    # # ----------- fill-between figures!!! -------------#
    # ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4)
