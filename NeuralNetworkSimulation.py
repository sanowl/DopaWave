from brian2 import *

# Start the scope
start_scope()

# Neuron parameters
tau = 10*ms
v_rest = -70*mV
v_reset = -65*mV
v_thresh = -50*mV
refractory_period = 5*ms

# Synapse parameters
weight = 0.1

# Define the neuron model
eqs = '''
dv/dt = (v_rest - v) / tau : volt (unless refractory)
'''

# Create a population of neurons
N = 100
neurons = NeuronGroup(N, eqs, threshold='v>v_thresh', reset='v=v_reset', refractory=refractory_period, method='exact')
neurons.v = v_rest

# Create a Poisson input to stimulate the neurons
poisson_input = PoissonGroup(N, rates=15*Hz)

# Create synapses between the input and the neurons
synapses = Synapses(poisson_input, neurons, on_pre='v += weight*mV')
synapses.connect()

# Record the spikes and membrane potential
spike_monitor = SpikeMonitor(neurons)
voltage_monitor = StateMonitor(neurons, 'v', record=0)

# Set up the live plotting
figure(figsize=(12, 6))
raster_ax = subplot(211)
voltage_ax = subplot(212)

raster_ax.set_title('Spike Raster Plot')
raster_ax.set_xlabel('Time (ms)')
raster_ax.set_ylabel('Neuron index')
raster_plot, = raster_ax.plot([], [], '.k')

voltage_ax.set_title('Membrane Potential of First Neuron')
voltage_ax.set_xlabel('Time (ms)')
voltage_ax.set_ylabel('Membrane potential (mV)')
voltage_plot, = voltage_ax.plot([], [])

ion()  # Turn on interactive mode

def update_plot():
    raster_plot.set_xdata(spike_monitor.t/ms)
    raster_plot.set_ydata(spike_monitor.i)
    
    voltage_plot.set_xdata(voltage_monitor.t/ms)
    voltage_plot.set_ydata(voltage_monitor.v[0]/mV)
    
    raster_ax.relim()
    raster_ax.autoscale_view()
    
    voltage_ax.relim()
    voltage_ax.autoscale_view()
    
    draw()
    pause(0.01)

# Run the simulation with live updates
for _ in range(100):
    run(10*ms)
    update_plot()

ioff()  # Turn off interactive mode
show()
