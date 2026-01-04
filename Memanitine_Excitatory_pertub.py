#!/usr/bin/env python3
"""
EXCITATORY SUPPRESSION MODEL
Simulates NMDA-mediated excitatory synaptic transmission
and applies a conductance reduction to model memantine effect.
"""

# Silence non-critical warnings to keep output clean
import warnings
warnings.filterwarnings('ignore')

# Print a clean header for readability
print("\n" + "=" * 70)
print("EXCITATORY SUPPRESSION MODEL")
print("=" * 70 + "\n")

# Import core simulation and analysis libraries
print("Importing libraries...")
import brian2
brian2.prefs.codegen.target = 'numpy'
from brian2 import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("✓ Libraries loaded successfully\n")

# Fix the random seed for reproducibility
seed(42)

# Define simulation time resolution
defaultclock.dt = 0.1 * ms

# -------------------------
# MODEL PARAMETERS
# -------------------------
print("=" * 70)
print("PARAMETERS")
print("=" * 70)

# Baseline NMDA synaptic conductance
g_NMDA_baseline = 1.0 * nS

# Fractional reduction applied to NMDA conductance
reduction_percent = 0.20

# NMDA conductance under memantine condition
g_NMDA_memantine = g_NMDA_baseline * (1 - reduction_percent)

print(f"Baseline NMDA conductance: {g_NMDA_baseline}")
print(f"Reduced NMDA conductance:  {g_NMDA_memantine} (-20%)")
print("=" * 70 + "\n")

# -------------------------
# MAGNESIUM BLOCK FUNCTION
# -------------------------
# Voltage-dependent Mg²⁺ block of NMDA receptors
@check_units(V=volt, result=1)
def mg_block(V):
    return 1.0 / (1.0 + (1.0 / 3.57) * exp(-0.062 * V / mV))

# -------------------------
# BASELINE SIMULATION
# -------------------------
print("[1/2] RUNNING BASELINE SIMULATION...\n")

# Single postsynaptic neuron receiving NMDA input
neuron_base = NeuronGroup(
    1,
    '''
    dv/dt = (-(v + 70*mV) + I_NMDA/nS) / (20*ms) : volt
    I_NMDA : amp
    ''',
    threshold='v > -50*mV',
    reset='v = -70*mV',
    method='euler'
)

# Initialize membrane potential
neuron_base.v = -70 * mV

# Define presynaptic spike times
spike_times = array([100, 300, 500, 700, 900]) * ms
stimulus_base = SpikeGeneratorGroup(1, zeros(5, dtype=int), spike_times)

# Synaptic dynamics for NMDA receptors
synapse_eqs = '''
I_NMDA_post = g_nmda * s * (v_post - 0*mV) * mg_block(v_post) : amp (summed)
ds/dt = alpha * (1 - s) - s / tau_decay : 1 (clock-driven)
alpha : hertz
tau_decay : second
g_nmda : siemens
'''

# Create synapse connecting stimulus to neuron
syn_base = Synapses(
    stimulus_base,
    neuron_base,
    model=synapse_eqs,
    on_pre='s += 1',
    method='euler'
)

syn_base.connect()
syn_base.g_nmda = g_NMDA_baseline
syn_base.alpha = 1 / (3 * ms)
syn_base.tau_decay = 100 * ms

# Record membrane voltage and NMDA current
mon_v_base = StateMonitor(neuron_base, 'v', record=True)
mon_I_base = StateMonitor(syn_base, 'I_NMDA_post', record=True)

# Run baseline simulation
print("  Running baseline simulation (1 s)...")
run(1 * second)
print("  ✓ Baseline complete\n")

# Extract baseline data
time = mon_v_base.t / ms
v_base = mon_v_base.v[0] / mV
I_base = mon_I_base.I_NMDA_post[0] / pA

# -------------------------
# MEMANTINE SIMULATION
# -------------------------
print("[2/2] RUNNING REDUCED NMDA SIMULATION...\n")

# Reset Brian2 internal state
start_scope()

# Recreate neuron
neuron_mem = NeuronGroup(
    1,
    '''
    dv/dt = (-(v + 70*mV) + I_NMDA/nS) / (20*ms) : volt
    I_NMDA : amp
    ''',
    threshold='v > -50*mV',
    reset='v = -70*mV',
    method='euler'
)

neuron_mem.v = -70 * mV

# Reuse identical stimulus
stimulus_mem = SpikeGeneratorGroup(1, zeros(5, dtype=int), spike_times)

# Create synapse with reduced NMDA conductance
syn_mem = Synapses(
    stimulus_mem,
    neuron_mem,
    model=synapse_eqs,
    on_pre='s += 1',
    method='euler'
)

syn_mem.connect()
syn_mem.g_nmda = g_NMDA_memantine
syn_mem.alpha = 1 / (3 * ms)
syn_mem.tau_decay = 100 * ms

# Record membrane voltage and NMDA current
mon_v_mem = StateMonitor(neuron_mem, 'v', record=True)
mon_I_mem = StateMonitor(syn_mem, 'I_NMDA_post', record=True)

# Run reduced-conductance simulation
print("  Running reduced NMDA simulation (1 s)...")
run(1 * second)
print("  ✓ Reduced NMDA simulation complete\n")

# Extract memantine data
v_mem = mon_v_mem.v[0] / mV
I_mem = mon_I_mem.I_NMDA_post[0] / pA

# -------------------------
# EPSP ANALYSIS
# -------------------------
print("=" * 70)
print("ANALYZING RESULTS")
print("=" * 70 + "\n")

peaks_base = []
peaks_mem = []

# Measure EPSP peak after each presynaptic spike
for spike_t in spike_times / ms:
    start_idx = int(spike_t * 10)
    end_idx = int((spike_t + 50) * 10)

    if end_idx < len(v_base):
        peaks_base.append(max(v_base[start_idx:end_idx]) + 70)
        peaks_mem.append(max(v_mem[start_idx:end_idx]) + 70)

# Compute averages and percentage reduction
avg_base = np.mean(peaks_base)
avg_mem = np.mean(peaks_mem)
reduction = ((avg_base - avg_mem) / avg_base) * 100

print(f"Baseline EPSP:   {avg_base:.2f} mV")
print(f"Reduced EPSP:    {avg_mem:.2f} mV")
print(f"Observed drop:   {reduction:.1f}%\n")

# -------------------------
# VISUALIZATION
# -------------------------
print("=" * 70)
print("GENERATING FIGURE")
print("=" * 70 + "\n")

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    'NMDA-Mediated Excitatory Suppression\n(-20% Conductance)',
    fontsize=15,
    fontweight='bold'
)

# Voltage trace
ax1 = plt.subplot(3, 1, 1)
ax1.plot(time, v_base, label='Baseline', lw=2.5)
ax1.plot(time, v_mem, label='Reduced NMDA', lw=2.5)
ax1.set_ylabel('Membrane Voltage (mV)')
ax1.set_title('Postsynaptic Voltage')
ax1.legend()
ax1.grid(True, alpha=0.3)

# NMDA current
ax2 = plt.subplot(3, 1, 2)
ax2.plot(time, I_base, label='Baseline', lw=2.5)
ax2.plot(time, I_mem, label='Reduced NMDA', lw=2.5)
ax2.set_ylabel('NMDA Current (pA)')
ax2.set_title('NMDA Synaptic Current')
ax2.legend()
ax2.grid(True, alpha=0.3)

# EPSP bar comparison
ax3 = plt.subplot(3, 1, 3)
x = np.arange(len(peaks_base))
width = 0.35

ax3.bar(x - width / 2, peaks_base, width, label='Baseline')
ax3.bar(x + width / 2, peaks_mem, width, label='Reduced NMDA')

ax3.set_ylabel('Peak EPSP (mV)')
ax3.set_xlabel('Stimulus')
ax3.set_title(f'EPSP Reduction: {reduction:.1f}%')
ax3.legend()
ax3.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

# Save output figure
filename = 'excitatory_suppression_results.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved as {filename}")

# Display plot
plt.show()

print("\n✓ Script finished successfully")
print("=" * 70)

