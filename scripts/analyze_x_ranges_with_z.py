import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

def analyze_and_visualize(left_data, right_data, z_offset=1885):
    """Analyze X ranges and visualize with adjustable Z offset"""
    # Filter for sprint frames
    left_sprint = left_data[
        (left_data['frame'] >= 44589) & 
        (left_data['frame'] <= 45372)
    ]
    right_sprint = right_data[
        (right_data['frame'] >= 46474) & 
        (right_data['frame'] <= 47162)
    ]
    
    print("\nX-coordinate ranges during sprints:")
    print(f"Left camera X range: {left_sprint['pitch_x'].min():.1f} - {left_sprint['pitch_x'].max():.1f}")
    print(f"Right camera X range: {right_sprint['pitch_x'].min():.1f} - {right_sprint['pitch_x'].max():.1f}")
    
    # Create 3D visualization
    fig = go.Figure()
    
    # Plot left camera tracks
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
    
    # Plot right camera tracks with z_offset
    for track_id in right_sprint['tracking_id'].unique():
        track = right_sprint[right_sprint['tracking_id'] == track_id].copy()
        # Apply the Z offset
        track['frame'] = track['frame'] - z_offset
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
        title=f"Sprint Comparison (Z offset: {z_offset} frames)",
        scene=dict(
            xaxis_title='Pitch X',
            yaxis_title='Pitch Y',
            zaxis_title='Frame',
            camera=dict(
                eye=dict(x=2, y=2, z=1.5)
            )
        ),
        showlegend=True
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
    
    # You can adjust this value to change the Z offset
    z_offset = 1885  # Default offset based on frame numbers
    
    fig = analyze_and_visualize(left_data, right_data, z_offset)
    fig.show() 