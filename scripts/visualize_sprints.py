import pandas as pd
import plotly.graph_objects as go
import os

def load_sprint_data(file_path, frame_range=None):
    """Load data for a specific frame range"""
    data = pd.read_csv(file_path)
    if frame_range:
        data = data[
            (data['frame'] >= frame_range[0]) & 
            (data['frame'] <= frame_range[1])
        ]
    return data

def create_sprint_visualization():
    # Sprint frame ranges from the conversation
    sprints = {
        'left': {
            'file': "../data/camL_1.csv",
            'frames': (44589, 45372),
            'color': 'blue'
        },
        'right': {
            'file': "../data/camR_1.csv",
            'frames': (46474, 47162),
            'color': 'red'
        }
    }
    
    fig = go.Figure()
    
    # Process each sprint
    for camera, info in sprints.items():
        if not os.path.exists(info['file']):
            print(f"File not found: {info['file']}")
            continue
            
        print(f"\nProcessing {camera} camera data...")
        print(f"Frame range: {info['frames']}")
        
        data = load_sprint_data(info['file'], info['frames'])
        
        # Plot each track in the sprint
        for track_id in data['tracking_id'].unique():
            track_data = data[data['tracking_id'] == track_id].copy()
            
            fig.add_trace(
                go.Scatter3d(
                    x=track_data['pitch_x'],
                    y=track_data['pitch_y'],
                    z=track_data['frame'],
                    mode='lines',
                    name=f"{camera} - Track {track_id}",
                    line=dict(color=info['color'], width=2),
                    opacity=0.6
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Sprint Comparison (Left Camera: 44589-45372, Right Camera: 46474-47162)",
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
        font=dict(color='white'),
        height=1000,
        width=1500
    )
    
    return fig

if __name__ == "__main__":
    fig = create_sprint_visualization()
    fig.show() 