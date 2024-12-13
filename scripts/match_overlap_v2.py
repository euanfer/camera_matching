import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

def find_best_z_offset(left_data, right_data, overlap_x_range=(463, 619)):
    """Find the best frame offset by matching patterns in overlap region"""
    
    # First filter for the sprint frames
    left_sprint = left_data[
        (left_data['frame'] >= 44589) & 
        (left_data['frame'] <= 45372)
    ]
    right_sprint = right_data[
        (right_data['frame'] >= 46474) & 
        (right_data['frame'] <= 47162)
    ]
    
    # Then filter for overlap region
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
    
    # Try different offsets within a reasonable range
    best_offset = 0
    best_match = float('inf')
    
    # We expect offset to be around 1885 frames (46474 - 44589)
    for offset in range(1800, 2000, 1):
        error = 0
        matches = 0
        
        # Compare player positions in overlap region
        for left_frame in left_overlap['frame'].unique():
            right_frame = left_frame + offset
            
            left_pos = left_overlap[left_overlap['frame'] == left_frame][['pitch_x', 'pitch_y']]
            right_pos = right_overlap[right_overlap['frame'] == right_frame][['pitch_x', 'pitch_y']]
            
            if len(left_pos) == 0 or len(right_pos) == 0:
                continue
            
            # Calculate minimum distances between points
            for _, left_point in left_pos.iterrows():
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

def visualize_matched_tracks(left_data, right_data, offset):
    fig = go.Figure()
    
    # Plot left camera data
    for track_id in left_data['tracking_id'].unique():
        track = left 