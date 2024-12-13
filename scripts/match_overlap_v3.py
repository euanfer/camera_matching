import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

def find_best_z_offset(left_data, right_data, overlap_x_range=(463, 619)):
    """Find the best frame offset using vectorized operations"""
    
    # First filter for the sprint frames only
    left_sprint = left_data[
        (left_data['frame'] >= 44589) & 
        (left_data['frame'] <= 45372)
    ].copy()
    
    right_sprint = right_data[
        (right_data['frame'] >= 46474) & 
        (right_data['frame'] <= 47162)
    ].copy()
    
    print(f"Filtered to sprint frames:")
    print(f"Left frames: {left_sprint['frame'].min()} - {left_sprint['frame'].max()}")
    print(f"Right frames: {right_sprint['frame'].min()} - {right_sprint['frame'].max()}")
    
    # Filter for overlap region
    left_overlap = left_sprint[
        (left_sprint['pitch_x'] >= overlap_x_range[0]) & 
        (left_sprint['pitch_x'] <= overlap_x_range[1])
    ]
    right_overlap = right_sprint[
        (right_sprint['pitch_x'] >= overlap_x_range[0]) & 
        (right_sprint['pitch_x'] <= overlap_x_range[1])
    ]
    
    print(f"\nPoints in overlap region:")
    print(f"Left camera: {len(left_overlap)}")
    print(f"Right camera: {len(right_overlap)}")
    
    # Create frame counts for quick matching
    left_counts = left_overlap.groupby('frame').size()
    right_counts = right_overlap.groupby('frame').size()
    
    # Try offsets in steps of 5 frames first
    best_offset = 1885  # Initial guess based on frame numbers
    best_match = float('inf')
    
    # Coarse search
    for offset in range(1800, 2000, 5):
        # Shift right frames by offset
        shifted_right = right_counts.index - offset
        
        # Find overlapping frames
        common_frames = left_counts.index.intersection(shifted_right)
        
        if len(common_frames) > 0:
            # Compare player counts
            error = abs(
                left_counts[common_frames] - 
                right_counts[common_frames + offset]
            ).mean()
            
            if error < best_match:
                best_match = error
                best_offset = offset
                print(f"New best offset: {offset} (error: {error:.2f})")
    
    # Fine search around best offset
    fine_best = best_offset
    fine_error = best_match
    
    for offset in range(best_offset - 4, best_offset + 5):
        shifted_right = right_counts.index - offset
        common_frames = left_counts.index.intersection(shifted_right)
        
        if len(common_frames) > 0:
            error = abs(
                left_counts[common_frames] - 
                right_counts[common_frames + offset]
            ).mean()
            
            if error < fine_error:
                fine_error = error
                fine_best = offset
                print(f"Fine tuned offset: {offset} (error: {error:.2f})")
    
    return fine_best

def visualize_matched_tracks(left_data, right_data, offset):
    # Filter for sprint frames
    left_sprint = left_data[
        (left_data['frame'] >= 44589) & 
        (left_data['frame'] <= 45372)
    ]
    right_sprint = right_data[
        (right_data['frame'] >= 46474) & 
        (right_data['frame'] <= 47162)
    ]
    
    fig = go.Figure()
    
    # Plot left camera data
    for track_id in left_sprint['tracking_id'].unique():
        track = left_sprint[left_sprint['tracking_id'] == track_id]
        fig.add_trace(
            go.Scatter3d(
                x=track['pitch_x'],
                y=track['pitch_y'],
                z=track['frame'],
                mode='lines',
                name=f'Left {track_id}',
                line=dict(color='blue'),
                opacity=0.6
            )
        )
    
    # Plot right camera data with offset
    for track_id in right_sprint['tracking_id'].unique():
        track = right_sprint[right_sprint['tracking_id'] == track_id].copy()
        track['frame'] = track['frame'] - offset
        fig.add_trace(
            go.Scatter3d(
                x=track['pitch_x'],
                y=track['pitch_y'],
                z=track['frame'],
                mode='lines',
                name=f'Right {track_id}',
                line=dict(color='red'),
                opacity=0.6
            )
        )
    
    fig.update_layout(
        title=f"Synchronized Tracks (offset: {offset} frames)",
        scene=dict(
            xaxis_title='Pitch X',
            yaxis_title='Pitch Y',
            zaxis_title='Frame'
        )
    )
    
    return fig

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    left_path = os.path.join(data_dir, 'camL_1.csv')
    right_path = os.path.join(data_dir, 'camR_1.csv')
    
    print("Loading data...")
    left_data = pd.read_csv(left_path)
    right_data = pd.read_csv(right_path)
    
    print("\nFinding best offset...")
    offset = find_best_z_offset(left_data, right_data)
    print(f"\nBest offset found: {offset} frames")
    
    print("\nCreating visualization...")
    fig = visualize_matched_tracks(left_data, right_data, offset)
    fig.show() 