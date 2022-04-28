#!/usr/bin/env python3

import brian2
from brian2 import NeuronGroup, SpikeGeneratorGroup, SpikeMonitor, StateMonitor, Synapses
from brian2 import mV, pA, pF, ms, second, Hz, Gohm
import brian2.numpy_ as np
import matplotlib.pyplot as plt
import os
from os.path import join


def poisson_generator(rate, t_lim, unit_ms=False, round_to=None, seed=None):
    """
    Draw events from a Poisson point process.

    Note: the implementation assumes a spike at t=t_lim[0], although this spike
    is not included in the spike list.

    inputs:
        rate                the rate of the discharge in Hz
        t_lim               tuple containing start and end time
        unit_ms             use ms as unit for times in t_lim and resulting
                            events
        rount_to            if passed, times are rounded to multiples of this
                            number
        seed                seed for random number generation

    returns:
        events              numpy array containing spike times in s (or ms, if
                            unit_ms is set)

    """

    assert len(t_lim) == 2

    if seed:
        np.random.seed(seed)

    if unit_ms:
        t_lim = (t_lim[0] / 1000, t_lim[1] / 1000)

    if rate > 0.:
        events_ = [t_lim[0]]

        while events_[-1] < t_lim[1]:
            T = t_lim[1] - events_[-1]

            # expected number of events
            num_expected = T * rate

            # number of events to generate
            num_generate = np.ceil(num_expected + 3 * np.sqrt(num_expected))
            num_generate = int(max(num_generate, 1000))

            beta = 1. / rate
            isi_ = np.random.exponential(beta, size=num_generate)
            newevents_ = np.cumsum(isi_) + events_[-1]
            events_ = np.append(events_, newevents_)

        lastind = np.searchsorted(events_, t_lim[1])
        events_ = events_[1:lastind]  # drop ghost spike at start

        if unit_ms:
            events_ *= 1000

    elif rate == 0:
        events_ = np.asarray([])

    else:
        raise ValueError('requested negative rate.')

    if round_to is not None:
        events_ = np.round(events_ / round_to) * round_to

    if seed:
        np.random.seed()

    return events_


def generate_stimulus(t_sim, rate, event_rate, jitter=0., sequence=False, num_neurons=200, num_event_neurons=100, dt=.1):
    """
    Generate the target spikes for the experiments: a list containing a list of
    spikes for each input neuron. The first 100 neurons are subject to events
    (synchronous input) with optional jitter or firing in a sequence.

    :param t_sim: simulation time
    :param rate: background firing rate of all neurons in Hz
    :param event_rate: rate of synchronized firing events in Hz
    :param jitter: standard deviation of jitter added to synchronous firing events in ms
    :param sequence: bool indicating whether neurons firing in a sequence
    :param num_neurons: number of input neurons
    :param num_event_neurons: number of input neurons which are subject to synchronized events
    :returns: np.ndarray with sender ids, np.ndarray with spike times
    """

    # generate background spikes
    spikes_ = [poisson_generator(rate, (dt, t_sim), unit_ms=True) for n in range(num_neurons)]

    # generate event spikes
    if event_rate > 0.:
        events_ = poisson_generator(event_rate, (0., t_sim), unit_ms=True)

        for n in range(num_event_neurons):
            # add sequence offset
            offs1 = n if sequence else 0.

            # add jitter
            offs2 = jitter * np.random.randn(*events_.shape) if jitter > 0. else 0.

            spikes_[n] = np.sort(np.append(spikes_[n], events_ + offs1 + offs2))

        # round to dt and sort out duplicate spikes
        spikes_ = [np.unique(np.round(spikes / dt)) * dt for spikes in spikes_]

        # clip values (negative times may occur due to jitter)
        spikes_ = [np.clip(spikes, dt, t_sim) for spikes in spikes_]

    # brian data format
    ids = np.concatenate([k * np.ones(len(sp), dtype=int) for k, sp in enumerate(spikes_)])
    times = np.concatenate(spikes_)
    assert len(times) == len(ids)

    return ids, times


def correlate_sets(spikes1_, spikes2_, binsize, corr_range, num_pairs=None, allow_same_indices=True, use_mp=True):
    """
    Computed a binned correlation between two groups of units where a sets of
    events is given for each unit.

    :param spikes1_: list containing lists of events for units in group 1
    :param spikes2_: list containing lists of events for units in group 2
    :param binsize: size of bins for correlation
    :param corr_range: tuple containing min and max offset for correlation
    :param num_pairs: number of pairs to use for correlation, if None, all pairs are used
    :param allow_same_indices: whether to allow computing correlations for the same index for both sets
    :param use_mp: whether to use multiprocessing
    """

    if num_pairs is None:
        ind1_ = np.arange(len(spikes1_))
        ind2_ = np.arange(len(spikes2_))
        pairs = np.asarray(np.meshgrid(ind1_, ind2_)).T.reshape(-1, 2).T
    else:
        ind1_ = np.random.randint(len(spikes1_), size=num_pairs)
        ind2_ = np.random.randint(len(spikes2_), size=num_pairs)
        pairs = np.vstack((ind1_, ind2_))

    if not allow_same_indices:
        for k, (x, y) in enumerate(zip(pairs[0,:], pairs[1,:])):
            if x == y:
                pairs[1,k] = (y + 1) % len(spikes2_)

    x_ = [spikes1_[k] for k in pairs[0,:]]
    y_ = [spikes2_[k] for k in pairs[1,:]]
    bs_ = [binsize] * len(x_)
    cr_ = [corr_range] * len(y_)
    args_ = zip(x_, y_, bs_, cr_)

    if use_mp:
        from joblib import Parallel, delayed
        import multiprocessing as mp
        r_ = Parallel(n_jobs=mp.cpu_count())(delayed(binned_correlation)(*args) for args in args_)
    else:
        r_ = [binned_correlation(*args) for args in args_]

    t_, corr_ = zip(*r_)
    t_ = t_[0]

    return t_, np.sum(corr_, axis=0)


def binned_correlation(t1_, t2_, binsize, corr_range):
    """
    Computed a binned correlation between two sets of events.

    :param t1_: times of events in set 1
    :param t2_: times of events in set 2
    :param binsize: size of bins for correlation
    :param corr_range: tuple containing min and max offset for correlation
    """
    assert len(corr_range) == 2

    dt_low, dt_high = corr_range
    t_ = np.arange(dt_low, dt_high + binsize, binsize)
    N_bins = len(t_)
    bin0 = int(np.round(N_bins / 2))

    corr_ = np.zeros_like(t_)

    for t2 in t2_:
        ind_low = np.searchsorted(t1_, t2 + dt_low)
        ind_high = np.searchsorted(t1_, t2 + dt_high)

        for t1 in t1_[ind_low:ind_high]:
            ind = int(np.round((t2 - t1) / binsize)) + bin0
            corr_[ind] += 1

    return t_, corr_


def experiment(*, sequence=False, jitter=0, alpha=1.1, t_sim=200, title='', plot_learning_window=False):
    """
    Perform a single STDP experiment.

    :param sequence: whether first 100 neurons should fired sequentially when events occur
    :param jitter: std of jitter for synchronized events in ms
    :param alpha: relative shape of negative part of STDP
    :param t_sim: simulation time in seconds
    :param title: base name for saving plots
    :param plot_learning_window: whether to visualize learning rule
    """

    # setup output directory

    outdir = 'out'
    outdir = join(os.path.dirname(__file__), outdir)
    os.makedirs(outdir, exist_ok=True)

    # network and simulation parameters

    num_input_with_events = 100  # blue neurons
    num_input_no_events = 100  # red neurons

    t_sim = t_sim * second

    # setup brian2

    net = brian2.Network()

    # neuron and synapse parameters
    r_bg = 8 * Hz
    r_event = 2 * Hz  # additional organized events for blue neurons
    theta = -55 * mV
    u_reset = -70 * mV
    delta_abs = 1 * ms
    tau_syn = 10 * ms
    R_m = 0.03 * Gohm
    C_m = 1500 * pF

    # setup neuron
    # note: use I_syn as name for the synaptic current (assumed below)

    lif_eqs = '''
    du/dt = ( -(u - u_rest) + R_m * I_syn(t)) / tau_m : volt (unless refractory)
    dI_syn/dt = - I_syn / tau_syn : ampere (unless refractory)
    '''

    blue_neurons = NeuronGroup(num_input_with_events, lif_eqs, threshold=theta, reset=u_reset, refractory=delta_abs, method='euler')
    red_neurons = NeuronGroup(num_input_no_events, lif_eqs, threshold=theta, reset=u_reset, refractory=delta_abs, method='euler')

    blue_spike_mon = SpikeMonitor(blue_neurons)
    red_spike_mon = SpikeMonitor(red_neurons)

    # setup inputs
    num_input = num_input_with_events + num_input_no_events

    stim_ids, stim_times = generate_stimulus(t_sim / ms, 8., 2., jitter, sequence, num_neurons=num_input)

    inputs = SpikeGeneratorGroup(num_input, stim_ids, stim_times)
    input_spike_mon = SpikeMonitor(inputs)

    # setup synapses

    synapses = Synapses(...)

    ...

    syn_state_mon = StateMonitor(synapses, ['w'], record=True, dt=1 * second)  # monitor the weights

    # TODO end
    # ----------------------------------------------------------------------

    # define units for main simulation

    units = [blue_neurons, red_neurons, blue_spike_mon, red_spike_mon, inputs, input_spike_mon, synapses, syn_state_mon]

    net.add(units)

    # ----------------------------------------------------------------------
    # record learning window

    if plot_learning_window:

        min_dt, max_dt = -100 * ms, 100 * ms  # limits of learning window measurement

        # setup network

        lw_net = brian2.Network()

        # setup stimulus

        _dt = brian2.defaultclock.dt
        lw_t_offset = _dt - min_dt  # offset relative to postsynaptic spike (dt = 0)

        # one neuron for each time step within learning window, each neuron
        # fires once
        lw_times = np.arange(lw_t_offset + min_dt, lw_t_offset + max_dt + _dt, _dt)
        num_lw_in = len(lw_times)
        lw_ids = np.arange(num_lw_in)

        lw_inputs = SpikeGeneratorGroup(num_lw_in, lw_ids, lw_times)

        # setup neuron

        # force neuron to fire at time dt = 0
        lw_neuron = NeuronGroup(1, 'I_syn : ampere\nt_spike : second', threshold='t >= t_spike and t < t_spike+dt')
        lw_neuron.t_spike = lw_t_offset
        lw_spike_mon = SpikeMonitor(lw_neuron)

        # setup synapses, using definitions from above

        lw_w0 = 50 * pA
        lw_synapses = Synapses(lw_inputs, lw_neuron, syn_eqs, on_pre=on_pre, on_post=on_post)
        lw_synapses.connect()
        lw_synapses.w = lw_w0

        # run

        lw_net.add([lw_inputs, lw_neuron, lw_spike_mon, lw_synapses])

        lw_net.run(max(lw_times) + _dt)

        assert len(lw_spike_mon) == 1
        assert lw_spike_mon.t == lw_t_offset

        # plot learning window

        plt.figure(figsize=(6, 4))

        ax = plt.gca()
        ax.plot((lw_times - lw_t_offset) / ms, (lw_synapses.w - lw_w0) / pA, lw=2, c='k')
        ax.set_xlim(min_dt / ms, max_dt / ms)
        ax.locator_params(nbins=3)
        ax.set_xlabel(r'$(t_\mathrm{pre} - t_\mathrm{post})$ / ms')
        ax.set_ylabel(r'$\Delta w$ / pA')
        ax.grid(axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if title:
            plt.savefig(join(outdir, title + '_learning_window.pdf'))

    # ----------------------------------------------------------------------
    # simulation 2: main experiment

    net.run(t_sim, report='stdout', report_period=1 * second)

    # analysis

    neuron_spikes = spike_mon.t

    t_ = np.arange(t_sim / second) * second
    neuron_rates = np.zeros_like(t_)

    for k, t1 in enumerate(t_[1:]):
        t_bin = t_[1] - t_[0]  # bin size
        t0 = t1 - t_bin
        neuron_rates[k+1] = sum((t0 < neuron_spikes) & (neuron_spikes <= t1)) / t_bin

    # spike sorting

    input_spike_trains = input_spike_mon.spike_trains()
    input_spikes = [input_spike_trains[i] for i in sorted(input_spike_trains.keys())]

    # weight statistics
    weights = syn_state_mon.w
    w_ev_mean = weights[:num_input_with_events,:].mean(axis=0)
    w_ev_std = weights[:num_input_with_events,:].std(axis=0)
    w_noev_mean = weights[num_input_with_events:,:].mean(axis=0)
    w_noev_std = weights[num_input_with_events:,:].std(axis=0)

    # crop spike times in order to save time during convolution
    t_max_spikes = 25 * second
    neuron_spikes = neuron_spikes[neuron_spikes<t_max_spikes]
    for k in range(len(input_spikes)):
        input_spikes[k] = input_spikes[k][input_spikes[k]<t_max_spikes]

    spikes_ev, spikes_noev = input_spikes[:num_input_with_events], input_spikes[num_input_with_events:]
    num_pairs = 200
    binsize = 5.
    corr_range = (-100., 100.)

    jkt_corr_, ii_corr_ev = correlate_sets(
            [sp / ms for sp in spikes_ev],
            [sp / ms for sp in spikes_ev],
            binsize=binsize,
            corr_range=corr_range,
            num_pairs=num_pairs,
            allow_same_indices=False)
    t_corr_, ii_corr_noev = correlate_sets(
            [sp / ms for sp in spikes_noev],
            [sp / ms for sp in spikes_noev],
            binsize=binsize,
            corr_range=corr_range,
            num_pairs=num_pairs,
            allow_same_indices=False)
    t_corr_, io_corr_ev = correlate_sets(
            [sp / ms for sp in spikes_ev],
            [neuron_spikes / ms],
            binsize=binsize,
            corr_range=corr_range)
    t_corr_, io_corr_noev = correlate_sets(
            [sp / ms for sp in spikes_noev],
            [neuron_spikes / ms],
            binsize=binsize,
            corr_range=corr_range)

    # plot

    t_max_plot = 5

    c_group1, c_group2 = ['#0343df', '#dc143c']

    plt.figure(figsize=(9, 3.5))

    ax = plt.gca()
    for n, spikes in enumerate(input_spikes):
        c = c_group1 if n < num_input_with_events else c_group2
        ax.scatter(spikes / second, n * np.ones_like(spikes), c=c, marker='.', s=1)
    ax.set_xlim(0, t_max_plot)
    ax.set_ylim(0, len(input_spikes))
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([num_input_with_events / 2, num_input_with_events + num_input_no_events / 2])
    ax.set_yticklabels(['group 1', 'group 2'], rotation=90)
    ax.set_xlabel(r'$t$ / s')
    ax.set_ylabel(r'input neuron')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.get_yticklabels()[0], color=c_group1, va='center')
    plt.setp(ax.get_yticklabels()[1], color=c_group2, va='center')
    plt.setp(ax.get_yticklines(), visible=False)

    plt.tight_layout()
    if title:
        plt.savefig(join(outdir, title + '_input_spikes.pdf'))

    errorbar_skip = int(t_sim / second / 10)

    plt.figure(figsize=(12, 3))
    plt.subplots_adjust(left=.07, top=.90, right=.99, bottom=.15, wspace=.5)

    ax = plt.subplot(1, 3, 1)
    im = ax.imshow(weights / pA, aspect='auto')
    ax.locator_params(nbins=5)
    ax.set_ylim(0, 200)
    ax.set_xlabel(r'$t$ / s')
    ax.set_ylabel(r'input neuron')
    ax.text(-.35, 1.02, 'A', fontsize=18, transform=ax.transAxes)
    cbar = plt.colorbar(im)
    cbar.set_label(r'$w$ / pA')

    ax = plt.subplot(1, 3, 2)
    ax.plot(t_ / second, w_ev_mean / pA, c=c_group1, lw=2, label='group 1')
    ax.plot(t_ / second, w_noev_mean / pA, c=c_group2, lw=2, label='group 2')
    e1 = ax.errorbar(t_[::errorbar_skip] / second, w_ev_mean[::errorbar_skip] / pA, w_ev_std[::errorbar_skip] / pA, c=c_group1, fmt='.', capsize=5, clip_on=False)
    e2 = ax.errorbar(t_[::errorbar_skip] / second, w_noev_mean[::errorbar_skip] / pA, w_noev_std[::errorbar_skip] / pA, c=c_group2, fmt='.', capsize=5, clip_on=False)
    ax.legend(loc='best')
    ax.autoscale(axis='x', tight=True)
    ax.locator_params(nbins=5)
    ax.set_xlabel(r'$t$ / s')
    ax.set_ylabel(r'mean weight')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-.27, 1.02, 'B', fontsize=18, transform=ax.transAxes)
    [b.set_clip_on(False) for b in e1[1] + e2[1] + e1[2] + e2[2]]

    ax = plt.subplot(1, 3, 3)
    ax.plot(t_[1:] / second, neuron_rates[1:] / Hz, c='k', lw=2)
    ax.autoscale(axis='x', tight=True)
    ax.locator_params(nbins=5)
    ax.set_xlabel(r'$t$ / s')
    ax.set_ylabel(r'neuron firing rate')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-.27, 1.02, 'C', fontsize=18, transform=ax.transAxes)

    if title:
        plt.savefig(join(outdir, title + '_weights.pdf'))

    fig = plt.figure(figsize=(9, 4))
    plt.subplots_adjust(left=.09, top=.91, right=.98, bottom=.12, hspace=1, wspace=.5)

    ax = plt.subplot(2, 2, 1)
    ax.plot(t_corr_, ii_corr_ev, marker='o', c=c_group1)
    ax.set_title('i/i corr. group 1')

    ax = plt.subplot(2, 2, 2)
    ax.plot(t_corr_, io_corr_ev, marker='o', c=c_group1)
    ax.set_title('i/o corr. group 1')

    ax = plt.subplot(2, 2, 3)
    ax.plot(t_corr_, ii_corr_noev, marker='o', c=c_group2)
    ax.set_title('i/i corr. group 2')

    ax = plt.subplot(2, 2, 4)
    ax.plot(t_corr_, io_corr_noev, marker='o', c=c_group2)
    ax.set_title('i/o corr. group 2')

    for k, ax in enumerate(fig.axes):
        ax.axvline(0, c='k', ls='--', alpha=.5, zorder=-1)
        ax.autoscale(axis='x', tight=True)
        ax.locator_params(nbins=4)
        ax.set_ylabel('count')
        ax.set_xlabel(r'$\Delta t$ / ms')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-.25, 1.1, chr(65 + k), fontsize=18, transform=ax.transAxes)

    if title:
        plt.savefig(join(outdir, title + '_correlations.pdf'))


if __name__ == '__main__':

    # run experiments

    experiment(sequence=False, jitter=0, title='task_a', plot_learning_window=True)

    ...

    plt.show() # avoid having multiple plt.show()s in your code
