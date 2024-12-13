import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def analyze_tracking_data(file_path, color):
    """Analyze and prepare data for visualization"""
    try:
        # First try UTF-8
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # Try with 'latin-1' encoding
            data = pd.read_csv(file_path, encoding='latin-1')
        except Exception as e:
            # If that fails, try to read with error handling
            data = pd.read_csv(file_path, encoding='utf-8', errors='replace')
    
    # Add debug print to see the data structure
    print(f"\nReading file: {file_path}")
    print("Columns:", data.columns.tolist())
    print("First few rows:")
    print(data.head())
    
    # Quality filtering parameters
    VELOCITY_THRESHOLD = 50
    MIN_TRACK_LENGTH = 30
    MAX_POSITION_JUMP = 50
    
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

def create_visualization():
    data_files = [
        "../data/camL_1.csv",
        "../data/camM_1.csv",
        "../data/camR_1.csv"
    ]
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}]
        ],
        subplot_titles=('3D Trajectories', 'Track Length Distribution', 'Active Tracks'),
        column_widths=[0.7, 0.3]
    )
    
    colors = {
        'camL_1.csv': 'blue',
        'camM_1.csv': 'red',
        'camR_1.csv': 'green'
    }
    
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        file_name = os.path.basename(file_path)
        color = colors[file_name]
        
        data = analyze_tracking_data(file_path, color)
        
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
                ),
                row=1, col=1
            )
        
        track_lengths = data.groupby('tracking_id').size()
        fig.add_trace(
            go.Histogram(
                x=track_lengths,
                name=f"{file_name} lengths",
                opacity=0.7,
                marker_color=color
            ),
            row=1, col=2
        )
        
        active_tracks = data.groupby('frame')['tracking_id'].nunique()
        fig.add_trace(
            go.Scatter(
                x=active_tracks.index,
                y=active_tracks.values,
                name=f"{file_name} active",
                line=dict(color=color)
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=1000,
        width=1500,
        title="Combined Track Analysis",
        showlegend=True,
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
    
    fig.update_scenes(
        bgcolor='black',
        xaxis=dict(title='X', gridcolor='gray', showbackground=False),
        yaxis=dict(title='Y', gridcolor='gray', showbackground=False),
        zaxis=dict(title='Frame', gridcolor='gray', showbackground=False),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    return fig

if __name__ == "__main__":
    fig = create_visualization()
    fig.show()