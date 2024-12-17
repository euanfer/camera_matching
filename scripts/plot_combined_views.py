import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def load_and_filter_data(file_path):
    """Load and filter data with the same parameters as analyze_tracking_data"""
    data = pd.read_csv(file_path)
    
    # Quality filtering parameters
    VELOCITY_THRESHOLD = 50
    MIN_TRACK_LENGTH = 30
    MAX_POSITION_JUMP = 50

    start_frame = 0
    end_frame = 1000

    data = data[(data['frame'] >= start_frame) & (data['frame'] <= end_frame)]

    valid_track_ids = []
    for track_id in data['tracking_id'].unique():
        track = data[data['tracking_id'] == track_id]
        track_length = len(track)
        max_velocity = track['velocity'].max() if 'velocity' in data.columns else 0
        x_jumps = abs(track['pitch_x'].diff()).max()
        y_jumps = abs(track['pitch_y'].diff()).max()
        
        is_valid = (
            track_length >= MIN_TRACK_LENGTH and
            max_velocity <= VELOCITY_THRESHOLD and
            x_jumps <= MAX_POSITION_JUMP and
            y_jumps <= MAX_POSITION_JUMP
        )
        if is_valid:
            valid_track_ids.append(track_id)
    
    return data[data['tracking_id'].isin(valid_track_ids)]

def create_combined_visualization():
    data_files = [
        "./data/combined/camL_1_kal.csv",
        "./data/combined/camM_1_kal.csv",
        "./data/combined/camR_1_kal.csv"
    ]
    print(f'{data_files[0]}')
    colors = {
        f'{data_files[0]}': 'blue',
        f'{data_files[1]}': 'red',
        f'{data_files[2]}': 'green'
    }
    
    # Create single plot figure
    fig = go.Figure()
    
    # Load and plot data from each camera
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        file_name = os.path.basename(file_path)
        color = colors[file_path]
        
        print(f"Processing {file_name}...")
        data = load_and_filter_data(file_path)
        
        # Plot each track
        for track_id in data['tracking_id'].unique():
            track_data = data[data['tracking_id'] == track_id].copy()
            track_data = track_data.iloc[::3]  # Downsample for performance
            
            fig.add_trace(
                go.Scatter3d(
                    x=track_data['pitch_x'],
                    y=track_data['pitch_y'],
                    z=track_data['frame'],
                    mode='lines',
                    name=f"{file_name} - Track {track_id}",
                    line=dict(color=color, width=2),
                    opacity=0.6
                )
            )
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1500,
        title="Combined Camera Views Analysis",
        scene=dict(
            bgcolor='black',
            xaxis=dict(title='Pitch X', gridcolor='gray', showbackground=False),
            yaxis=dict(title='Pitch Y', gridcolor='gray', showbackground=False),
            zaxis=dict(title='Frame', gridcolor='gray', showbackground=False),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
    
    return fig

if __name__ == "__main__":
    fig = create_combined_visualization()
    fig.show() 