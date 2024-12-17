import numpy as np
from filterpy.kalman import KalmanFilter
import pandas as pd
from tqdm import tqdm
import networkx as nx

def calculate_measurement_noise(camera_position, player_position):
    """
    Calculate measurement noise based on distance from camera
    
    Args:
        camera_position: (x, y) tuple of camera position
        player_position: (x, y) tuple of player position
    """
    distance = np.linalg.norm(np.array(player_position) - np.array(camera_position))
    
    # Base noise level
    base_noise = 10
    
    # Scale noise quadratically with distance
    # (uncertainty grows with square of distance)
    noise_scale = (distance / 10.0) ** 2.4  # normalize by 100 units
    
    return base_noise * noise_scale

def create_kalman_filter(camera_position, initial_position):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # State transition matrix (unchanged)
    dt = 1.0
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Measurement matrix (unchanged)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # Dynamic measurement noise
    noise = calculate_measurement_noise(camera_position, initial_position)
    kf.R = np.eye(2) * noise
    
    # Process noise (unchanged)
    q = 0.1
    kf.Q = np.array([
        [q*dt**4/4, 0, q*dt**3/2, 0],
        [0, q*dt**4/4, 0, q*dt**3/2],
        [q*dt**3/2, 0, q*dt**2, 0],
        [0, q*dt**3/2, 0, q*dt**2]
    ])
    
    kf.P *= 1000
    return kf

def smooth_track(track_data, camera_position):
    positions = np.column_stack((track_data['pitch_x'], track_data['pitch_y']))
    smoothed_positions = np.zeros_like(positions)
    
    # Create Kalman filter with initial position
    kf = create_kalman_filter(camera_position, positions[0])
    kf.x = np.array([positions[0,0], positions[0,1], 0, 0])
    
    # Forward pass
    for i, pos in enumerate(positions):
        # Update measurement noise for current position
        noise = calculate_measurement_noise(camera_position, pos)
        kf.R = np.eye(2) * noise
        
        kf.predict()
        kf.update(pos)
        smoothed_positions[i] = kf.x[:2]
    
    return smoothed_positions

def process_all_tracks(df, camera_position):
    df_smoothed = df.copy()
    
    for track_id in tqdm(df['tracking_id'].unique(), desc="Processing tracks"):
        track_mask = df['tracking_id'] == track_id
        track_data = df[track_mask]
        
        if len(track_data) > 1:
            smoothed_positions = smooth_track(track_data, camera_position)
            df_smoothed.loc[track_mask, 'pitch_x'] = smoothed_positions[:, 0]
            df_smoothed.loc[track_mask, 'pitch_y'] = smoothed_positions[:, 1]
    
    return df_smoothed



# Camera positions (example values - adjust these to your actual camera positions)

if __name__ == "__main__":
    df_paths = [
        'data/raw/camL_1_raw_fixed.csv',
        'data/raw/camM_1_raw_fixed.csv',
        'data/raw/camR_1_raw_fixed.csv'
    ]

    start_frame = 0
    end_frame = 1000

    for df_path in tqdm(df_paths, desc="Processing files"):
        df = pd.read_csv(df_path)
        df = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)]
        # Determine camera position from filename
        camera_id = df_path.split('/')[-1][:4]  # extracts 'camL', 'camM', or 'camR'
        camera_position = (525,0)
        
        # Apply Kalman smoothing with camera-specific measurement noise
        df_smoothed = process_all_tracks(df, camera_position)
        
        output_path = f'data/raw/{df_path.split("/")[-1].replace("_raw_fixed.csv", "_kal.csv")}'
        df_smoothed.to_csv(output_path, index=False)
