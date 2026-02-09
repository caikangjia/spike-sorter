import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os

def load_neural_data(filepath):
    """
    Loads neural data from a MATLAB .mat file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        mat_contents = scipy.io.loadmat(filepath)
        plot_data = mat_contents['plot_data'][0, 0]
        
        # Extract fields
        data = {
            'mean_lead': plot_data['r_seq_mean_LEAD'],
            'mean_trail': plot_data['r_seq_mean_TRAIL'],
            'sem_lead': plot_data['r_seq_sem_LEAD'],
            'sem_trail': plot_data['r_seq_sem_TRAIL'],
            'phase_seq': plot_data['phase_seq'],
            'cells2plot': plot_data['cells2plot']
        }
        
        # Verify shapes
        print("Data Loaded:")
        print(f"  LEAD Mean Shape: {data['mean_lead'].shape}")
        print(f"  TRAIL Mean Shape: {data['mean_trail'].shape}")
        
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading/parsing .mat file: {e}")

def preprocess_data(data):
    """
    Concatenates LEAD and TRAIL data and applies Z-score normalization per neuron.
    """
    features_raw = np.hstack([data['mean_lead'], data['mean_trail']])
    scaler = StandardScaler()
    features_z = scaler.fit_transform(features_raw.T).T
    
    print(f"Feature Vector Shape (concatenated & z-scored): {features_z.shape}")
    return features_z, features_raw

def plot_dendrogram(features, save_path=None):
    """
    Computes and plots hierarchical clustering dendrogram.
    """
    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Neuron Index")
    plt.ylabel("Distance")
    
    linkage_matrix = linkage(features, method='ward')
    dendrogram(linkage_matrix, no_labels=True)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Dendrogram to {save_path}")
    plt.show()
    plt.close()

def run_kmeans_analysis(features, k_values=[3, 4, 5, 6]):
    """
    Runs KMeans for multiple K values and calculates Silhouette Scores.
    """
    best_score = -1
    best_k = k_values[0]
    best_labels = None
    results = {}

    print("\n--- K-Means Clustering Results ---")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        results[k] = {'score': score, 'labels': labels}
        print(f"  K={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            
    print(f"Best K based on Silhouette Score: {best_k}")
    return best_k, best_labels, results

def visualize_clusters(data, features_z, labels, k, save_dir):
    """
    Generates Heatmaps, Profile Plots, and PCA plots.
    """
    # Sort data by cluster labels
    sorted_indices = np.argsort(labels)
    sorted_features = features_z[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # 1. Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(sorted_features, cmap='viridis', center=0, cbar_kws={'label': 'Z-Scored Firing Rate'})
    
    # Draw lines to separate clusters
    current_idx = 0
    for cluster_id in range(k):
        count = np.sum(labels == cluster_id)
        current_idx += count
        plt.axhline(current_idx, color='white', linewidth=1)
        
    # Draw vertical line separating LEAD and TRAIL
    midpoint = data['mean_lead'].shape[1]
    plt.axvline(midpoint, color='white', linestyle='--', linewidth=2)
    plt.text(midpoint/2, -5, 'LEAD', ha='center', color='black', fontsize=12, fontweight='bold')
    plt.text(midpoint + midpoint/2, -5, 'TRAIL', ha='center', color='black', fontsize=12, fontweight='bold')
    
    plt.title(f'Neural Firing Patterns (Sorted by Cluster, K={k})')
    plt.xlabel('Time Points')
    plt.ylabel('Neurons')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'heatmap_k{k}.png'), dpi=300)
    plt.show()
    plt.close()
    
    # 2. Average Profiles
    time_len = data['mean_lead'].shape[1]
    t = np.arange(time_len)
    if 'phase_seq' in data and data['phase_seq'].size == time_len:
         t = data['phase_seq'].flatten()

    plt.figure(figsize=(14, 4 * ((k+1)//2)))
    
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        n_neurons = len(cluster_indices)
        
        # Get raw data for this cluster
        lead_mean_cluster = data['mean_lead'][cluster_indices]
        trail_mean_cluster = data['mean_trail'][cluster_indices]
        
        pop_mean_lead = np.mean(lead_mean_cluster, axis=0)
        pop_sem_lead = np.std(lead_mean_cluster, axis=0) / np.sqrt(n_neurons)
        
        pop_mean_trail = np.mean(trail_mean_cluster, axis=0)
        pop_sem_trail = np.std(trail_mean_cluster, axis=0) / np.sqrt(n_neurons)
        
        # Plot
        plt.subplot((k + 1) // 2, 2, i + 1)
        
        # LEAD
        plt.plot(t, pop_mean_lead, label='LEAD', color='blue')
        plt.fill_between(t, pop_mean_lead - pop_sem_lead, pop_mean_lead + pop_sem_lead, color='blue', alpha=0.2)
        
        # TRAIL
        plt.plot(t, pop_mean_trail, label='TRAIL', color='red')
        plt.fill_between(t, pop_mean_trail - pop_sem_trail, pop_mean_trail + pop_sem_trail, color='red', alpha=0.2)
        
        plt.title(f'Cluster {i+1} (n={n_neurons})')
        plt.xlabel('Time/Phase')
        plt.ylabel('Firing Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cluster_profiles_k{k}.png'), dpi=300)
    plt.show()
    plt.close()

    # 3. PCA Scatter
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(features_z)
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='w', s=60)
    plt.title(f'PCA of Neuronal Activity (K={k})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'pca_scatter_k{k}.png'), dpi=300)
    plt.show()
    plt.close()
    
    print(f"Saved plots for K={k} in {save_dir}")

def generate_schematics(data, labels, k, save_dir):
    """
    Generates simplified schematic diagrams for each cluster with annotations.
    """
    time_len = data['mean_lead'].shape[1]
    t = np.arange(time_len)
    if 'phase_seq' in data and data['phase_seq'].size == time_len:
         t = data['phase_seq'].flatten()

    plt.figure(figsize=(5 * k, 6))
    
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0: 
            continue
            
        # Get mean profiles
        lead_mean = np.mean(data['mean_lead'][cluster_indices], axis=0)
        trail_mean = np.mean(data['mean_trail'][cluster_indices], axis=0)
        
        # Smooth for schema look
        sigma = 3
        lead_smooth = gaussian_filter1d(lead_mean, sigma)
        trail_smooth = gaussian_filter1d(trail_mean, sigma)
        
        ax = plt.subplot(1, k, i + 1)
        
        # Plot thick lines
        ax.plot(t, lead_smooth, color='#1f77b4', linewidth=4, label='LEAD', alpha=0.9)
        ax.plot(t, trail_smooth, color='#d62728', linewidth=4, label='TRAIL', alpha=0.9)
        
        # Shading Logic
        # 1. Shade where TRAIL > LEAD significantly
        diff = trail_smooth - lead_smooth
        threshold = 0.2 * np.max(np.abs(np.concatenate([lead_smooth, trail_smooth])))
        fill_mask = diff > threshold
        ax.fill_between(t, lead_smooth, trail_smooth, where=fill_mask, 
                        color='#d62728', alpha=0.2, interpolate=True)
                        
        # 2. Shade peaks in LEAD if they are distinct
        lead_max = np.max(lead_smooth)
        if lead_max > 0:
            peak_mask = lead_smooth > (0.7 * lead_max)
            ax.fill_between(t, 0, lead_smooth, where=peak_mask, 
                            color='#1f77b4', alpha=0.1, interpolate=True)

        # Automated Annotations
        full_concat = np.concatenate([lead_smooth, trail_smooth])
        global_max = np.max(full_concat) if len(full_concat) > 0 else 1.0
        
        # Check bias
        lead_auc = np.sum(lead_smooth)
        trail_auc = np.sum(trail_smooth)
        total_auc = lead_auc + trail_auc
        
        annotations = []
        if total_auc > 0:
            bias = (lead_auc - trail_auc) / total_auc
            if bias > 0.2:
                annotations.append("LEAD Dominant")
            elif bias < -0.2:
                annotations.append("TRAIL Dominant")
            else:
                annotations.append("Balanced Response")
                
        # Check timing (Early vs Late)
        mid_idx = len(t) // 2
        early_act = np.mean(lead_smooth[:mid_idx]) + np.mean(trail_smooth[:mid_idx])
        late_act = np.mean(lead_smooth[mid_idx:]) + np.mean(trail_smooth[mid_idx:])
        
        if early_act > 1.5 * late_act:
            annotations.append("Early Onset")
        elif late_act > 1.5 * early_act:
            annotations.append("Late/Sustained")
            
        # Add text
        y_pos = global_max * 1.1
        for idx, note in enumerate(annotations):
            ax.text(t[len(t)//2], y_pos - (idx * 0.1 * global_max), note, 
                    ha='center', fontsize=12, fontweight='bold', color='#333333')
            
        # Minimalist Style
        ax.set_title(f'Cluster {i+1}', fontsize=16, pad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(width=2, labelsize=10)
        
        if i == 0:
            ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
            ax.legend(frameon=False, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cluster_schematics_k{k}.png'), dpi=300)
    plt.show()
    plt.close()
    print(f"Saved Schematics for K={k} in {save_dir}")

def main():
    FILE_PATH = r'D:\M021\peth_plot_data.mat'
    OUTPUT_DIR = 'clustering_results'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Starting analysis on {FILE_PATH}...")
    
    # 1. Load Data
    try:
        data = load_neural_data(FILE_PATH)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        return

    # 2. Preprocess
    features_z, features_raw = preprocess_data(data)
    
    # 3. Hierarchical Clustering (Dendrogram)
    plot_dendrogram(features_z, save_path=os.path.join(OUTPUT_DIR, 'dendrogram.png'))
    
    # 4. K-Means Analysis
    k_range = [3, 4, 5, 6]
    best_k, best_labels, all_results = run_kmeans_analysis(features_z, k_range)
    
    # 5. Visualize Best Clustering
    print(f"\nGenerating plots for Best K={best_k}...")
    visualize_clusters(data, features_z, best_labels, best_k, OUTPUT_DIR)
    
    # 6. Generate Schematics
    print(f"Generating schematics for Best K={best_k}...")
    generate_schematics(data, best_labels, best_k, OUTPUT_DIR)

    print("\nAnalysis Complete.")
    print(f"All results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()