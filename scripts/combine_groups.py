import pandas as pd
import numpy as np

data_paths = ['data/raw/camL_1_kal.csv', 'data/raw/camM_1_kal.csv', 'data/raw/camR_1_kal.csv']
group_paths = ['data/groups/camL.csv', 'data/groups/camM.csv', 'data/groups/camR.csv']

for data_path, group_path in zip(data_paths, group_paths):
    # Read the data
    df = pd.read_csv(data_path)
    groups = pd.read_csv(group_path)

    # For each group, set all tracking_ids to match the first entry
    for group in groups:
        if len(group) > 1:  # Only process groups with multiple tracks
            primary_id = group[0]  # Use first track_id as the primary
            df.loc[df['tracking_id'].isin(group), 'tracking_id'] = primary_id
    
    # Sort by tracking_id and frame for interpolation
    df = df.sort_values(['tracking_id', 'frame'])
    
    # Interpolate missing frames for each track
    df = df.set_index('frame').groupby('tracking_id').apply(
        lambda group: group.reindex(
            range(int(group.index.min()), int(group.index.max()) + 1)
        ).interpolate(method='linear')
    ).reset_index(level=1)

    # Save the processed data
    df.to_csv(f'data/combined/{data_path.split("/")[-1]}', index=False)
    