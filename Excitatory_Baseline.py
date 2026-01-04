#!/usr/bin/env python3
"""
Excitatory baseline network (Brian2)

Outputs:
- exc_only_raster.png
- exc_only_rate.png
"""

import warnings
warnings.filterwarnings("ignore")

import brian2 as b2
from brian2 import ms, mV, Hz, second, nA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Avoid cython messages
b2.prefs.codegen.target = "numpy"


# ----------------------------
# NETWORK SIZE
# ----------------------------
N_E = 100


# ----------------------------
# NEURON MODEL (LIF)
# ----------------------------
tau = 20 * ms
v_rest = -65 * mV
v_reset = -65 * mV
v_th = -55 * mV
refrac = 3 * ms

tau_ext = 10 * ms  # external current decay

eqs = """
dI_ext/dt = -I_ext/tau_ext : amp
dv/dt = (v_rest - v + ((I_rec + I_ext) / nA)*mV) / tau : volt
I_rec : amp
"""

E = b2.NeuronGroup(
    N_E,
    model=eqs,
    threshold="v > v_th",
    reset="v = v_reset",
    refractory=refrac,
    method="euler",
)

E.v = v_rest
E.I_ext = 0 * nA
E.I_rec = 0 * nA


# ----------------------------
# RECURRENT EXCITATORY SYNAPSES (E->E)
# ----------------------------
Erev_exc = 0 * mV
tau_exc = 5 * ms

syn_exc = """
dg/dt = -g/tau_exc : 1 (clock-driven)
w : 1
I_rec_post = (g*w)*(Erev_exc - v_post) * (nA/mV) : amp (summed)
"""

S_EE = b2.Synapses(E, E, model=syn_exc, on_pre="g += 1")
S_EE.connect(p=0.05)
S_EE.w = 0.25


# ----------------------------
# EXTERNAL DRIVE (Poisson spikes -> I_ext)
# ----------------------------
P = b2.PoissonGroup(N_E, rates=20 * Hz)
S_PE = b2.Synapses(P, E, on_pre="I_ext_post += 2.5*nA")
S_PE.connect(j="i")


# ----------------------------
# MONITORS
# ----------------------------
M = b2.SpikeMonitor(E)
R = b2.PopulationRateMonitor(E)


# ----------------------------
# RUN
# ----------------------------
b2.run(1 * second)

print("Total spikes:", M.num_spikes)


# ----------------------------
# PLOTS
# ----------------------------
plt.figure(figsize=(12, 5))
plt.plot(M.t / ms, M.i, ".", markersize=2)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron index")
plt.title("Excitatory-only raster")
plt.xlim(0, 1000)
plt.ylim(-1, N_E)
plt.tight_layout()
plt.savefig("exc_only_raster.png", dpi=200)

plt.figure(figsize=(12, 4))
plt.plot(R.t / ms, R.smooth_rate(window="flat", width=50 * ms) / Hz)
plt.xlabel("Time (ms)")
plt.ylabel("Hz")
plt.title("Excitatory-only population rate")
plt.xlim(0, 1000)
plt.tight_layout()
plt.savefig("exc_only_rate.png", dpi=200)


