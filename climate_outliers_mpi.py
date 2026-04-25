from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore') 

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define the exact features for impacts
    features = [
        "avg_temperature_c", "temperature_change_c", "sea_level_rise_mm", 
        "heatwave_days", "rainfall_change_mm", "flood_risk", "drought_risk"
    ]
    
    NUM_CLUSTERS = 4      # Number of standard historical weather patterns
    NUM_OUTLIER_CLUSTERS = 3 # Number of new extreme weather patterns to find in 2070
    SIGMA_MULTIPLIER = 2  # Threshold multiplier 

    historical_centroids = None
    thresholds = None

  
    # 1: BASELINE ESTABLISHMENT (Master Only)
    
    if rank == 0:
        print(f"--- Master (Rank 0): Initializing Pipeline with {size} processes ---")
        
        # 1. Load the combined data
        print("Loading climatedata_2070.csv...")
        df = pd.read_csv("climatedata_2070.csv")
        
        # 2. Split into Historical vs 2070
        hist_df = df[df['type'] == 'Historical'][features]
        syn_df = df[df['type'] == 'Synthetic_2070'][features]
        
        # 3. Normalize the data for Euclidean distance
        scaler = MinMaxScaler()
        hist_scaled = scaler.fit_transform(hist_df)
        syn_scaled = scaler.transform(syn_df) 
        
        # 4. Find the Historical Baselines using K-Means
        print("Clustering Historical Baselines...")
        kmeans_hist = KMeans(n_clusters=NUM_CLUSTERS, random_state=42).fit(hist_scaled)
        historical_centroids = kmeans_hist.cluster_centers_
        
        # 5. Calculate boundary thresholds for each cluster
        labels = kmeans_hist.labels_
        thresholds = np.zeros(NUM_CLUSTERS)
        
        for i in range(NUM_CLUSTERS):
            cluster_points = hist_scaled[labels == i]
            distances = np.linalg.norm(cluster_points - historical_centroids[i], axis=1)
            # Threshold = Mean distance + (3 * Standard Deviation)
            thresholds[i] = np.mean(distances) + (SIGMA_MULTIPLIER * np.std(distances))
            
        print("Historical baselines established.")
        
        # Prepare the data to scatter (convert to contiguous C-array for MPI)
        syn_scaled = np.ascontiguousarray(syn_scaled, dtype=np.float64)
        total_rows = syn_scaled.shape[0]
        
    else:
        syn_scaled = None
        total_rows = 0

    # 2: BROADCAST & SCATTER
    # Send baselines and thresholds to all workers
    historical_centroids = comm.bcast(historical_centroids, root=0)
    thresholds = comm.bcast(thresholds, root=0)
    total_rows = comm.bcast(total_rows, root=0)

    # Calculate chunk sizes for Scatter
    chunk_size = total_rows // size
    local_data = np.empty((chunk_size, len(features)), dtype=np.float64)

    # Scatter the 2070 dataset equally among all ranks
    if rank == 0:
        # Trim to fit exactly for Scatter (simplified for this assignment)
        syn_scaled = syn_scaled[:chunk_size * size] 
        
    comm.Scatter(syn_scaled, local_data, root=0)

    # 3: PARALLEL OUTLIER DETECTION
    local_outliers = []
    
    # Each node calculates Euclidean distance for its chunk
    for event in local_data:
        # Distance from this 2070 day to all historical centroids
        distances = np.linalg.norm(historical_centroids - event, axis=1)
        closest_cluster = np.argmin(distances)
        min_distance = distances[closest_cluster]
        
        # If the distance exceeds the historical threshold, it's a NEW pattern
        if min_distance > thresholds[closest_cluster]:
            local_outliers.append(event)

    local_outliers = np.array(local_outliers)
    print(f"Worker {rank}: Processed {chunk_size} rows, found {len(local_outliers)} outliers.")

    # STEP 4: GATHER & RE-CLUSTER
    
    gathered_outliers = comm.gather(local_outliers, root=0)

    if rank == 0:
        # Filter out empty arrays
        valid_outliers = [out for out in gathered_outliers if out.size > 0]
        
        if not valid_outliers:
            print("\n--- Master: Total 2070 Outliers Found = 0 ---")
            print("No extreme anomalies detected. Try lowering the SIGMA_MULTIPLIER.")
        else:
            # Stack the arrays if not empty
            all_outliers = np.vstack(valid_outliers)
            print(f"\n--- Master: Total 2070 Outliers Found = {len(all_outliers)} ---")
            
            if len(all_outliers) >= NUM_OUTLIER_CLUSTERS:
                print("Clustering Outliers to find Civil Engineering Hazards...")
                kmeans_outliers = KMeans(n_clusters=NUM_OUTLIER_CLUSTERS, n_init='auto', random_state=42).fit(all_outliers)
                
                # Inverse transform to get values back
                new_hazards = scaler.inverse_transform(kmeans_outliers.cluster_centers_)
                
                print("\n*** THE NEW NORMAL: 2070 CLIMATE ANOMALIES ***")
                for i, hazard in enumerate(new_hazards):
                    print(f"\nNew Regime {i+1}:")
                    for j, feature in enumerate(features):
                        print(f"  - {feature}: {hazard[j]:.2f}")
            else:
                print(f"Only found {len(all_outliers)} outliers. Not enough to group into {NUM_OUTLIER_CLUSTERS} clusters.")

if __name__ == "__main__":
    main()