from brian2 import *

# Start the scope
start_scope()

# Simulation parameters
duration = 1*second

# Neuron parameters
tau = 10*ms
v_rest = -70*mV
v_reset = -65*mV
v_thresh = -50*mV
refractory_period = 5*ms

# Synapse parameters
weight_exc = 0.1
weight_inh = -0.1

# Define the neuron model
eqs = '''
dv/dt = (v_rest - v) / tau : volt (unless refractory)
'''

# Create a population of neurons (80% excitatory, 20% inhibitory)
N_exc = 80
N_inh = 20
neurons_exc = NeuronGroup(N_exc, eqs, threshold='v>v_thresh', reset='v=v_reset', refractory=refractory_period, method='exact')
neurons_inh = NeuronGroup(N_inh, eqs, threshold='v>v_thresh', reset='v=v_reset', refractory=refractory_period, method='exact')

neurons_exc.v = v_rest
neurons_inh.v = v_rest

# Create Poisson inputs to stimulate the neurons
poisson_input_exc = PoissonGroup(N_exc, rates=15*Hz)
poisson_input_inh = PoissonGroup(N_inh, rates=15*Hz)

# Create synapses
synapses_exc = Synapses(poisson_input_exc, neurons_exc, on_pre='v += weight_exc*mV')
synapses_inh = Synapses(poisson_input_inh, neurons_inh, on_pre='v += weight_inh*mV')

synapses_exc.connect()
synapses_inh.connect()

# Record the spikes and membrane potential
spike_monitor_exc = SpikeMonitor(neurons_exc)
spike_monitor_inh = SpikeMonitor(neurons_inh)
voltage_monitor_exc = StateMonitor(neurons_exc, 'v', record=0)
voltage_monitor_inh = StateMonitor(neurons_inh, 'v', record=0)

# Set up the live plotting
figure(figsize=(12, 8))
raster_ax = subplot(311)
voltage_ax_exc = subplot(312)
voltage_ax_inh = subplot(313)

raster_ax.set_title('Spike Raster Plot')
raster_ax.set_xlabel('Time (ms)')
raster_ax.set_ylabel('Neuron index')
raster_plot_exc, = raster_ax.plot([], [], '.r', label='Excitatory')
raster_plot_inh, = raster_ax.plot([], [], '.b', label='Inhibitory')
raster_ax.legend()

voltage_ax_exc.set_title('Membrane Potential of First Excitatory Neuron')
voltage_ax_exc.set_xlabel('Time (ms)')
voltage_ax_exc.set_ylabel('Membrane potential (mV)')
voltage_plot_exc, = voltage_ax_exc.plot([], [], 'r')

voltage_ax_inh.set_title('Membrane Potential of First Inhibitory Neuron')
voltage_ax_inh.set_xlabel('Time (ms)')
voltage_ax_inh.set_ylabel('Membrane potential (mV)')
voltage_plot_inh, = voltage_ax_inh.plot([], [], 'b')

ion()  # Turn on interactive mode

def update_plot():
    # Update raster plot
    raster_plot_exc.set_xdata(spike_monitor_exc.t/ms)
    raster_plot_exc.set_ydata(spike_monitor_exc.i)
    raster_plot_inh.set_xdata(spike_monitor_inh.t/ms)
    raster_plot_inh.set_ydata(spike_monitor_inh.i)
    
    # Update voltage plots
    voltage_plot_exc.set_xdata(voltage_monitor_exc.t/ms)
    voltage_plot_exc.set_ydata(voltage_monitor_exc.v[0]/mV)
    voltage_plot_inh.set_xdata(voltage_monitor_inh.t/ms)
    voltage_plot_inh.set_ydata(voltage_monitor_inh.v[0]/mV)
    
    # Adjust the view
    raster_ax.relim()
    raster_ax.autoscale_view()
    
    voltage_ax_exc.relim()
    voltage_ax_exc.autoscale_view()
    
    voltage_ax_inh.relim()
    voltage_ax_inh.autoscale_view()
    
    draw()
    pause(0.01)

# Run the simulation with live updates
for _ in range(int(duration/(10*ms))):
    run(10*ms)
    update_plot()

ioff()  # Turn off interactive mode
show()
