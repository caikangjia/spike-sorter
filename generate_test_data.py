"""
Generate dummy neural data for testing the clustering pipeline.
"""
import numpy as np
import scipy.io

def generate_test_data(n_neurons=201, n_timepoints=100, filename='test_data.mat'):
    """
    Generates synthetic neural firing rate data with realistic patterns.
    
    Parameters:
    -----------
    n_neurons : int
        Number of neurons to simulate (default: 201)
    n_timepoints : int
        Number of time bins (default: 100)
    filename : str
        Output filename (default: 'test_data.mat')
    """
    
    print(f"Generating test data: {n_neurons} neurons × {n_timepoints} timepoints")
    
    # Time axis
    phase_seq = np.linspace(-np.pi, np.pi, n_timepoints)
    
    # Create 4 cluster types with distinct patterns
    cluster_size = n_neurons // 4
    
    # Cluster 1: LEAD dominant, early onset
    lead_1 = np.exp(-((phase_seq + 1.5)**2) / 0.5) * 30 + np.random.randn(n_timepoints) * 2
    trail_1 = np.ones(n_timepoints) * 5 + np.random.randn(n_timepoints) * 2
    
    # Cluster 2: TRAIL dominant, sustained
    lead_2 = np.ones(n_timepoints) * 8 + np.random.randn(n_timepoints) * 2
    trail_2 = 15 + 10 * (1 + np.tanh(phase_seq)) + np.random.randn(n_timepoints) * 2
    
    # Cluster 3: Balanced response
    lead_3 = 20 * np.exp(-(phase_seq**2) / 1.0) + np.random.randn(n_timepoints) * 2
    trail_3 = 18 * np.exp(-(phase_seq**2) / 1.0) + np.random.randn(n_timepoints) * 2
    
    # Cluster 4: Late onset
    lead_4 = np.exp(-((phase_seq - 1.0)**2) / 0.5) * 25 + np.random.randn(n_timepoints) * 2
    trail_4 = np.exp(-((phase_seq - 1.2)**2) / 0.5) * 28 + np.random.randn(n_timepoints) * 2
    
    # Combine clusters
    r_seq_mean_LEAD = np.vstack([
        np.tile(lead_1, (cluster_size, 1)) + np.random.randn(cluster_size, n_timepoints) * 3,
        np.tile(lead_2, (cluster_size, 1)) + np.random.randn(cluster_size, n_timepoints) * 3,
        np.tile(lead_3, (cluster_size, 1)) + np.random.randn(cluster_size, n_timepoints) * 3,
        np.tile(lead_4, (n_neurons - 3*cluster_size, 1)) + np.random.randn(n_neurons - 3*cluster_size, n_timepoints) * 3
    ])
    
    r_seq_mean_TRAIL = np.vstack([
        np.tile(trail_1, (cluster_size, 1)) + np.random.randn(cluster_size, n_timepoints) * 3,
        np.tile(trail_2, (cluster_size, 1)) + np.random.randn(cluster_size, n_timepoints) * 3,
        np.tile(trail_3, (cluster_size, 1)) + np.random.randn(cluster_size, n_timepoints) * 3,
        np.tile(trail_4, (n_neurons - 3*cluster_size, 1)) + np.random.randn(n_neurons - 3*cluster_size, n_timepoints) * 3
    ])
    
    # Ensure non-negative firing rates
    r_seq_mean_LEAD = np.maximum(r_seq_mean_LEAD, 0)
    r_seq_mean_TRAIL = np.maximum(r_seq_mean_TRAIL, 0)
    
    # Generate SEM (10% of mean)
    r_seq_sem_LEAD = r_seq_mean_LEAD * 0.1
    r_seq_sem_TRAIL = r_seq_mean_TRAIL * 0.1
    
    # Create plot_data structure
    plot_data = np.zeros(1, dtype=[
        ('r_seq_mean_LEAD', 'O'),
        ('r_seq_mean_TRAIL', 'O'),
        ('r_seq_sem_LEAD', 'O'),
        ('r_seq_sem_TRAIL', 'O'),
        ('phase_seq', 'O'),
        ('cells2plot', 'O')
    ])
    
    plot_data['r_seq_mean_LEAD'][0] = r_seq_mean_LEAD
    plot_data['r_seq_mean_TRAIL'][0] = r_seq_mean_TRAIL
    plot_data['r_seq_sem_LEAD'][0] = r_seq_sem_LEAD
    plot_data['r_seq_sem_TRAIL'][0] = r_seq_sem_TRAIL
    plot_data['phase_seq'][0] = phase_seq.reshape(1, -1)
    plot_data['cells2plot'][0] = np.arange(1, n_neurons + 1)
    
    # Save to .mat file
    scipy.io.savemat(filename, {'plot_data': plot_data})
    
    print(f"✓ Test data saved to: {filename}")
    print(f"  - {n_neurons} neurons with 4 distinct cluster patterns")
    print(f"  - {n_timepoints} time points")
    print(f"\nTo use this data, update FILE_PATH in clustering_analysis.py:")
    print(f"  FILE_PATH = r'{filename}'")

if __name__ == "__main__":
    generate_test_data()