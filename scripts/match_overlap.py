import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

def find_best_z_offset(left_data, right_data, overlap_x_range=(290, 320)):
    """Find the best frame offset by matching patterns in overlap region"""
    
    # Filter for overlap region
    left_overlap = left_data[
        (left_data['pitch_x'] >= overlap_x_range[0]) & 
        (left_data['pitch_x'] <= overlap_x_range[1])
    ]
    right_overlap = right_data[
        (right_data['pitch_x'] >= overlap_x_range[0]) & 
        (right_data['pitch_x'] <= overlap_x_range[1])
    ]
    
    # Get unique frames
    left_frames = sorted(left_overlap['frame'].unique())
    right_frames = sorted(right_overlap['frame'].unique())
    
    # Try different offsets within a reasonable range
    best_offset = 0
    best_match = float('inf')
    
    # We expect offset to be around 1885 frames based on the sprint data
    for offset in range(1800, 2000, 1):
        error = 0
        matches = 0
        
        # Compare player positions in overlap region
        for left_frame in left_frames:
            right_frame = left_frame + offset
            
            if right_frame not in right_frames:
                continue
                
            left_pos = left_overlap[left_overlap['frame'] == left_frame][['pitch_x', 'pitch_y']]
            right_pos = right_overlap[right_overlap['frame'] == right_frame][['pitch_x', 'pitch_y']]
            
            # Calculate minimum distances between points
            for _, left_point in left_pos.iterrows():
                if len(right_pos) > 0:
                    min_dist = min(
                        np.sqrt(((left_point - right_point)**2).sum()) 
                        for _, right_point in right_pos.iterrows()
                    )
                    error += min_dist
                    matches += 1
        
        if matches > 0:
            avg_error = error / matches
            if avg_error < best_match:
                best_match = avg_error
                best_offset = offset
                print(f"New best offset: {offset} (error: {avg_error:.2f})")
    
    return best_offset

def visualize_matched_data(left_data, right_data, z_offset):
    fig = go.Figure()
    
    # Plot left camera data in blue
    for track_id in left_data['tracking_id'].unique():
        track = left_data[left_data['tracking_id'] == track_id]
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
    
    # Plot right camera data in red with offset
    for track_id in right_data['tracking_id'].unique():
        track = right_data[right_data['tracking_id'] == track_id].copy()
        track['frame'] = track['frame'] - z_offset  # Apply the offset
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
        title=f"Synchronized Tracks (offset: {z_offset} frames)",
        scene=dict(
            bgcolor='black',
            xaxis=dict(title='Pitch X', gridcolor='gray'),
            yaxis=dict(title='Pitch Y', gridcolor='gray'),
            zaxis=dict(title='Frame', gridcolor='gray')
        ),
        showlegend=True,
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
    
    return fig

if __name__ == "__main__":
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    left_path = os.path.join(data_dir, 'camL_1.csv')
    right_path = os.path.join(data_dir, 'camR_1.csv')
    
    print("Loading data...")
    left_data = pd.read_csv(left_path)
    right_data = pd.read_csv(right_path)
    
    print("Finding best offset...")
    z_offset = find_best_z_offset(left_data, right_data)
    print(f"\nBest offset found: {z_offset} frames")
    
    print("Creating visualization...")
    fig = visualize_matched_data(left_data, right_data, z_offset)
    fig.show() 