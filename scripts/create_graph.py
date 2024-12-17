import numpy as np
from filterpy.kalman import KalmanFilter
import networkx as nx
import pandas as pd

class TrackNode:
    def __init__(self, track_id, frame, position, team_id, is_start=True):
        self.track_id = track_id
        self.frame = frame
        self.position = position  # (x, y)
        self.team_id = team_id
        self.is_start = is_start  # True for start node, False for end node
        self.kalman_predictions = []  # Store future predictions

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
    noise_scale = (distance / 10.0) ** 2.2  # normalize by 100 units
    
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

def generate_predictions(kf, initial_position, track_data, n_steps=30):
    """
    Generate n future position predictions using Kalman filter
    
    Args:
        kf: KalmanFilter object already initialized
        initial_position: (x, y) tuple of last known position
        track_data: DataFrame containing the track's positions
        n_steps: number of future steps to predict
    
    Returns:
        predictions: numpy array of shape (n_steps, 2) containing predicted positions
    """
    # Calculate velocity from last few positions
    if len(track_data) >= 60:
        last_pos = track_data.iloc[-1][['pitch_x', 'pitch_y']].values
        prev_pos = track_data.iloc[-10][['pitch_x', 'pitch_y']].values
        velocity = (last_pos - prev_pos) / 60
    else:
        last_pos = track_data.iloc[-1][['pitch_x', 'pitch_y']].values
        first_pos = track_data.iloc[0][['pitch_x', 'pitch_y']].values
        
        velocity = (last_pos - first_pos) / len(track_data)
        speed = np.linalg.norm(velocity)
        if speed > 3.3:
            velocity = velocity / speed * 0.06
    # Initialize state with last known position and calculated velocity
    kf.x = np.array([
        initial_position[0],  # x position
        initial_position[1],  # y position
        velocity[0],         # x velocity
        velocity[1]          # y velocity
    ])
    
    # Storage for predictions
    predictions = np.zeros((n_steps, 2))
    
    # Make n predictions
    for i in range(n_steps):
        # Predict next state
        kf.predict()
        
        # Store predicted position
        predictions[i] = kf.x[:2]
    
    return predictions

def calculate_edge_weight(node1, node2, max_frames=30):
    # Only connect end nodes to start nodes
    if node1.is_start or not node2.is_start:
        return float('inf')
    
    frame_diff = node2.frame - node1.frame
    if frame_diff <= 0 or frame_diff > max_frames:
        return float('inf')
    
    # Different teams should not connect
    if node1.team_id != node2.team_id:
        return float('inf')
    
    # Spatial distance
    spatial_dist = np.linalg.norm(
        np.array(node1.kalman_predictions[int(frame_diff-1)]) - 
        np.array(node2.position)
    )
    
    # Weight combines frame difference and spatial distance
    weight = spatial_dist * (1 + frame_diff/max_frames)
    
    return weight

def build_track_graph(tracks_df):
    G = nx.DiGraph()
    nodes = []
    
    # Create nodes for each track start/end
    for track_id in tracks_df['tracking_id'].unique():
        track = tracks_df[tracks_df['tracking_id'] == track_id]
        
        # Start node
        start_pos = (track.iloc[0]['pitch_x'], track.iloc[0]['pitch_y'])
        start_node = TrackNode(track_id, track.iloc[0]['frame'], 
                             start_pos, track.iloc[0]['team_id'], 
                             is_start=True)
        
        # End node with Kalman predictions
        end_pos = (track.iloc[-1]['pitch_x'], track.iloc[-1]['pitch_y'])
        end_node = TrackNode(track_id, track.iloc[-1]['frame'], 
                           end_pos, track.iloc[-1]['team_id'], 
                           is_start=False)
        
        # Generate Kalman predictions from end node
        kf = create_kalman_filter((525,0 ), end_pos)  # Your existing Kalman setup
        predictions = generate_predictions(kf, end_pos, track, n_steps=30)
        end_node.kalman_predictions = predictions
        
        nodes.extend([start_node, end_node])
        G.add_node(start_node)
        G.add_node(end_node)
    
    # Create edges between compatible nodes
    for node1 in nodes:
        for node2 in nodes:
            weight = calculate_edge_weight(node1, node2)
            if weight != float('inf'):
                G.add_edge(node1, node2, weight=weight)
    
    return G

def link_tracks(G):
    # Convert directed graph to undirected
    G_undirected = G.to_undirected()
    
    # Find minimum weight matching in bipartite graph
    matching = nx.min_weight_matching(G_undirected)
    
    # Create track groups based on matching
    track_groups = []
    used_tracks = set()
    
    for node1, node2 in matching:
        if node1.track_id not in used_tracks:
            current_group = [node1.track_id]
            used_tracks.add(node1.track_id)
            
            if node2.track_id not in used_tracks:
                current_group.append(node2.track_id)
                used_tracks.add(node2.track_id)
            
            track_groups.append(current_group)

    track_groups = [[int(x) for x in group] for group in track_groups]
    return track_groups



if __name__ == "__main__":

    df_paths = ['data/raw/camL_1_raw_fixed.csv', 'data/raw/camM_1_raw_fixed.csv', 'data/raw/camR_1_raw_fixed.csv']
    for df_path in df_paths:
        df = pd.read_csv(df_path)

        start_frame = 0
        end_frame = 1000
        df = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)]

        G = build_track_graph(df)
        track_groups = link_tracks(G)
        pd.DataFrame(track_groups).to_csv(f'data/groups/{df_path.split("/")[-1].split("_")[0]}.csv', index=False)
        print(track_groups)