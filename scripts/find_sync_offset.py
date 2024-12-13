import pandas as pd
import numpy as np
from scipy.signal import correlate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Get the absolute path to the data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), 'data')

def find_sync_offset(left_data, right_data, overlap_x_range=(290, 320)):
    """Find the frame offset that best aligns tracks in the overlap region"""
    
    # Filter for tracks in the overlap region
    left_overlap = left_data[
        (left_data['pitch_x'] >= overlap_x_range[0]) & 
        (left_data['pitch_x'] <= overlap_x_range[1])
    ]
    right_overlap = right_data[
        (right_data['pitch_x'] >= overlap_x_range[0]) & 
        (right_data['pitch_x'] <= overlap_x_range[1])
    ]
    
    # Create time series of player counts in overlap region
    left_counts = left_overlap.groupby('frame')['tracking_id'].nunique()
    right_counts = right_overlap.groupby('frame')['tracking_id'].nunique()
    
    # Pad series to same length
    max_frame = max(left_counts.index.max(), right_counts.index.max())
    min_frame = min(left_counts.index.min(), right_counts.index.min())
    all_frames = range(min_frame, max_frame + 1)
    
    left_series = pd.Series(0, index=all_frames)
    right_series = pd.Series(0, index=all_frames)
    
    left_series.update(left_counts)
    right_series.update(right_counts)
    
    # Find offset using cross-correlation
    correlation = correlate(left_series, right_series)
    lags = np.arange(-(len(right_series)-1), len(left_series))
    offset = lags[np.argmax(correlation)]
    
    return offset

def visualize_sync_comparison(left_data, right_data, offset):
    """Visualize tracks before and after synchronization"""
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Before Sync', 'After Sync'),
                       specs=[[{'type': 'scene'}, {'type': 'scene'}]])
    
    # Before sync
    for track_id in left_data['tracking_id'].unique():
        track = left_data[left_data['tracking_id'] == track_id]
        fig.add_trace(
            go.Scatter3d(x=track['pitch_x'], y=track['pitch_y'], z=track['frame'],
                        mode='lines', name=f'Left {track_id}',
                        line=dict(color='blue'), opacity=0.6),
            row=1, col=1
        )
    
    for track_id in right_data['tracking_id'].unique():
        track = right_data[right_data['tracking_id'] == track_id]
        fig.add_trace(
            go.Scatter3d(x=track['pitch_x'], y=track['pitch_y'], z=track['frame'],
                        mode='lines', name=f'Right {track_id}',
                        line=dict(color='red'), opacity=0.6),
            row=1, col=1
        )
    
    # After sync
    for track_id in left_data['tracking_id'].unique():
        track = left_data[left_data['tracking_id'] == track_id]
        fig.add_trace(
            go.Scatter3d(x=track['pitch_x'], y=track['pitch_y'], z=track['frame'],
                        mode='lines', name=f'Left {track_id}',
                        line=dict(color='blue'), opacity=0.6),
            row=1, col=2
        )
    
    for track_id in right_data['tracking_id'].unique():
        track = right_data[right_data['tracking_id'] == track_id].copy()
        track['frame'] = track['frame'] - offset  # Apply sync offset
        fig.add_trace(
            go.Scatter3d(x=track['pitch_x'], y=track['pitch_y'], z=track['frame'],
                        mode='lines', name=f'Right {track_id}',
                        line=dict(color='red'), opacity=0.6),
            row=1, col=2
        )
    
    fig.update_layout(height=800, width=1600, title=f"Sync Offset: {offset} frames")
    return fig

if __name__ == "__main__":
    # Load data using absolute paths
    left_path = os.path.join(data_dir, 'camL_1.csv')
    right_path = os.path.join(data_dir, 'camR_1.csv')
    
    print(f"Loading data from:")
    print(f"Left camera: {left_path}")
    print(f"Right camera: {right_path}")
    
    # Load data for the specific sprint frames
    left_data = pd.read_csv(left_path)
    right_data = pd.read_csv(right_path)
    
    print(f"\nLeft data shape: {left_data.shape}")
    print(f"Right data shape: {right_data.shape}")
    
    # Filter for sprint frames
    left_sprint = left_data[
        (left_data['frame'] >= 44589) & 
        (left_data['frame'] <= 45372)
    ]
    right_sprint = right_data[
        (right_data['frame'] >= 46474) & 
        (right_data['frame'] <= 47162)
    ]
    
    print(f"\nProcessing sprints:")
    print(f"Left sprint frames: {left_sprint['frame'].min()} - {left_sprint['frame'].max()}")
    print(f"Right sprint frames: {right_sprint['frame'].min()} - {right_sprint['frame'].max()}")
    
    offset = find_sync_offset(left_sprint, right_sprint)
    print(f"\nFound sync offset: {offset} frames")
    
    fig = visualize_sync_comparison(left_sprint, right_sprint, offset)
    fig.show() 