import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np

def filter_by_roi(df, min_x, max_x):
    """Filter dataframe to only include points within ROI"""
    return df[(df['pitch_x'] >= min_x) & (df['pitch_x'] <= max_x)]

def set_time_range(df, start_frame, end_frame):
    """Set time range of dataframe"""
    return df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)]

def show_all_tracks():
    plt.figure(figsize=(8, 8))
    plt.grid(True)
    
    # Plot target track with directional arrows
    plt.plot(target_track['pitch_x'], target_track['pitch_y'], 'k-->', 
             markevery=10, label=f'Target Track {target_id}')

    for track_id in track_ids2:
        track = df2[df2['tracking_id'] == track_id]
        plt.plot(track['pitch_x'], track['pitch_y'], '-->', 
                markevery=10, label=f'Track {track_id}')

    plt.legend()
    plt.show()

def find_closest_point(target_point, comparison_track):
    """Find closest point in comparison track to target track edge"""
    distances = comparison_track.apply(
        lambda row: euclidean(
            [target_point['pitch_x'], target_point['pitch_y']], 
            [row['pitch_x'], row['pitch_y']]
        ), 
        axis=1
    )

    closest_idx = distances.idxmin()
    matching_point = comparison_track.loc[closest_idx]

    return matching_point

def compare_tracks(track1, track2, close_threshold=2):
    """
    Compare tracks by:
    1. Finding last position in track1: P1(F1)
    2. Finding closest position in track2: P2(F2)
    3. Stepping backwards frame by frame from both points
    Returns similarity score (lower is better)
    """
    # Get last frame of track1
    track1_sorted = track1.sort_values('frame', ascending=False)
    last_point = track1_sorted.iloc[0]  # P1(F1)
    F1 = last_point['frame']
    
    # Find closest spatial match in track2
    distances = track2.apply(
        lambda row: euclidean(
            [last_point['pitch_x'], last_point['pitch_y']], 
            [row['pitch_x'], row['pitch_y']]
        ), 
        axis=1
    )
    closest_idx = distances.idxmin()
    matching_point = track2.loc[closest_idx]  # P2(F2)
    F2 = matching_point['frame']

    frame_diff = F1 - F2
    
    total_dist = 0
    count = 0
    close = True
    # Step backwards through frames
    while close:
        # Get points at current offset
        p1 = track1[track1['frame'] == (F1 - count)]
        p2 = track2[track2['frame'] == (F2 - count)]
        
        if p1.empty or p2.empty:
            continue
            
        # Calculate distance between points
        dist = euclidean(
            [p1.iloc[0]['pitch_x'], p1.iloc[0]['pitch_y']],
            [p2.iloc[0]['pitch_x'], p2.iloc[0]['pitch_y']]
        )
        if dist > close_threshold:
            close = False
        total_dist += dist
        count += 1
    
    return total_dist / (count * 1.2) if count > 0 else float('inf'), frame_diff

def is_near_edge_of_track(target_track, comparison_track, distance_threshold=30, frame_threshold=90):
    """Check if track is within threshold of edge of target track"""
    comparison_id = comparison_track['tracking_id'].iloc[0]

    target_start = target_track.iloc[0]
    target_end = target_track.iloc[-1]

    start_match = find_closest_point(target_start, comparison_track)
    end_match = find_closest_point(target_end, comparison_track)

    start_dist = euclidean([target_start['pitch_x'], target_start['pitch_y']], [start_match['pitch_x'], start_match['pitch_y']])
    end_dist = euclidean([target_end['pitch_x'], target_end['pitch_y']], [end_match['pitch_x'], end_match['pitch_y']])

    start_frame_diff = target_start['frame'] - start_match['frame']
    end_frame_diff = target_end['frame'] - end_match['frame']
    #print(f'{comparison_id} match start frame', start_match['frame'])

    if start_dist < distance_threshold and abs(start_frame_diff) < frame_threshold:
       # print(f'Distance to track {comparison_id} start of target track:', start_dist)
      #  print(f'Frame difference to track {comparison_id} start of target track:', start_frame_diff)
        return 'start_match', start_frame_diff

    if end_dist < distance_threshold and abs(end_frame_diff) < frame_threshold:
     #   print(f'Distance to track {comparison_id} end of target track:', end_dist)
     #   print(f'Frame difference to track {comparison_id} end of target track:', end_frame_diff)
        return 'end_match', end_frame_diff

    return False, None

def get_direction_vector(track, start_frame, end_frame):
    """
    Calculate direction vector between two points in a track
    Returns: (dx, dy) normalized direction vector
    """
    # Add debugging prints
    #print(f"\nDebug get_direction_vector:")
    #print(f"Looking for frames {start_frame} and {end_frame}")
    
    # Get points at specified frames
    start_point = track[track['frame'] == start_frame]
    end_point = track[track['frame'] == end_frame]
   # print(f'Tracking ID: {track["tracking_id"].iloc[0]}')
   # print('Track length:', len(track))
    # Check if points exist and print results
    if start_point.empty:
        print(f"Start point not found for frame {start_frame}")
    if end_point.empty:
        print(f"End point not found for frame {end_frame}")
    
    if start_point.empty or end_point.empty:
        return None
    
    # Print found points
   # print(f"Start point: {start_point.iloc[0][['frame', 'pitch_x', 'pitch_y']].to_dict()}")
   # print(f"End point: {end_point.iloc[0][['frame', 'pitch_x', 'pitch_y']].to_dict()}")
    
    # Calculate direction vector
    dx = end_point.iloc[0]['pitch_x'] - start_point.iloc[0]['pitch_x']
    dy = end_point.iloc[0]['pitch_y'] - start_point.iloc[0]['pitch_y']
    
    # Normalize vector
    magnitude = (dx**2 + dy**2)**0.5
    if magnitude == 0:
        print("Magnitude is zero - same start and end point")
        return None
        
    return (dx/magnitude, dy/magnitude)

def direction_analysis(target_track, comparison_track, frame_diff, match_type, direction_step=10):
    """
    Compare direction vectors of target and comparison tracks
    Returns: angle between vectors in degrees
    """
    # Add debugging print
    #print(f"\nDebug direction_analysis:")
    #print(f"Match type: {match_type}, Frame diff: {frame_diff}, Direction step: {direction_step}")
    
    comp_final_frame = comparison_track['frame'].iloc[-1]
    comp_first_frame = comparison_track['frame'].iloc[0]
    
    if match_type == 'start_match':
        target_start_frame = target_track['frame'].iloc[0]
        comp_start_frame = target_start_frame - frame_diff

        direction_step = min(10, comp_final_frame - comp_start_frame)

        target_end_frame = target_start_frame + direction_step
        comp_end_frame = comp_start_frame + direction_step
    else:  # end_match
        target_end_frame = target_track['frame'].iloc[-1]
        comp_end_frame = target_end_frame - frame_diff

        direction_step = min(10, comp_end_frame - comp_first_frame)

        target_start_frame = target_track['frame'].iloc[-1] - direction_step
        comp_start_frame = comp_end_frame - direction_step
    
    target_vector = get_direction_vector(target_track, target_start_frame, target_end_frame)
    comp_vector = get_direction_vector(comparison_track, comp_start_frame, comp_end_frame)
    
    if target_vector is None or comp_vector is None:
        return 179
    
    # Calculate angle between vectors using dot product
    dot_product = target_vector[0] * comp_vector[0] + target_vector[1] * comp_vector[1]
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def get_velocity(track, start_frame, end_frame):
    """
    Calculate velocity between two points in a track
    Returns: (vx, vy) velocity vector
    """
    # Get points at specified frames
    start_point = track[track['frame'] == start_frame]
    end_point = track[track['frame'] == end_frame]
    
    # Check if points exist
    if start_point.empty or end_point.empty:
        return (999,999)
    
    # Calculate displacement
    dx = end_point.iloc[0]['pitch_x'] - start_point.iloc[0]['pitch_x']
    dy = end_point.iloc[0]['pitch_y'] - start_point.iloc[0]['pitch_y']
    dt = end_frame - start_frame
    
    # Calculate velocity components
    vx = dx / dt
    vy = dy / dt
        
    return (vx, vy)

def velocity_analysis(target_track, comparison_track, frame_diff, match_type, direction_step=10):
    """
    Compare velocity vectors of target and comparison tracks
    Returns: velocity difference magnitude (lower is better)
    """

    comp_final_frame = comparison_track['frame'].iloc[-1]
    comp_first_frame = comparison_track['frame'].iloc[0]
    
    if match_type == 'start_match':
        target_start_frame = target_track['frame'].iloc[0]
        comp_start_frame = target_start_frame - frame_diff

        direction_step = min(10, comp_final_frame - comp_start_frame)

        target_end_frame = target_start_frame + direction_step
        comp_end_frame = comp_start_frame + direction_step
    else:  # end_match
        target_end_frame = target_track['frame'].iloc[-1]
        comp_end_frame = target_end_frame - frame_diff

        direction_step = min(10, comp_end_frame - comp_first_frame)

        target_start_frame = target_track['frame'].iloc[-1] - direction_step
        comp_start_frame = comp_end_frame + direction_step
    
    target_vel = get_velocity(target_track, target_start_frame, target_end_frame)
    comp_vel = get_velocity(comparison_track, comp_start_frame, comp_end_frame)
    
    if target_vel is None or comp_vel is None:
        return None
    
    # Calculate velocity difference magnitude
    vel_diff = ((target_vel[0] - comp_vel[0])**2 + 
                (target_vel[1] - comp_vel[1])**2)**0.5
    return vel_diff

def check_track_proximity_duration(target_track, comparison_track, frame_diff, match_type, distance_threshold=30):
    """
    Check how long tracks stay close together starting from match point
    Returns: number of frames tracks stay within threshold distance
    """
    if match_type == 'start_match':
        # Start from beginning of tracks and move forward
        target_frames = target_track['frame'].sort_values()
        comp_frames = comparison_track['frame'].sort_values()
    else:  # end_match
        # Start from end of tracks and move backward
        target_frames = target_track['frame'].sort_values(ascending=False)
        comp_frames = comparison_track['frame'].sort_values(ascending=False)
    
    close_frames = 0
    for target_frame in target_frames:
        comp_frame = target_frame - frame_diff
        
        # Get points at current frames
        target_point = target_track[target_track['frame'] == target_frame]
        comp_point = comparison_track[comparison_track['frame'] == comp_frame]
        
        if target_point.empty or comp_point.empty:
            break
            
        # Calculate distance between points
        distance = euclidean(
            [target_point.iloc[0]['pitch_x'], target_point.iloc[0]['pitch_y']],
            [comp_point.iloc[0]['pitch_x'], comp_point.iloc[0]['pitch_y']]
        )
        
        if distance > distance_threshold:
            break
            
        close_frames += 1
    
    return close_frames

def check_track_divergence(target_track, comparison_track, frame_diff, match_type):
    """
    Check how much tracks diverge by tracking maximum distance between them
    Returns: maximum distance between tracks during their overlap
    """
    if match_type == 'start_match':
        # Start from beginning of tracks and move forward
        target_frames = target_track['frame'].sort_values()
    else:  # end_match
        # Start from end of tracks and move backward
        target_frames = target_track['frame'].sort_values(ascending=False)
    
    distances = []
    for target_frame in target_frames:
        comp_frame = target_frame - frame_diff
        
        # Get points at current frames
        target_point = target_track[target_track['frame'] == target_frame]
        comp_point = comparison_track[comparison_track['frame'] == comp_frame]
        
        if target_point.empty or comp_point.empty:
            break
            
        # Calculate distance between points
        distance = euclidean(
            [target_point.iloc[0]['pitch_x'], target_point.iloc[0]['pitch_y']],
            [comp_point.iloc[0]['pitch_x'], comp_point.iloc[0]['pitch_y']]
        )
        
        distances.append(distance)
    
    return max(distances) if distances else float('inf')

df1 = pd.read_csv('csvs/cam1_1.csv')
df2 = pd.read_csv('csvs/cam2_1.csv')

target_id = 517
x_min, x_max = 0, 1050
start_frame, end_frame = 9000, 15000

angle_threshold = 30
velocity_threshold = 3
min_proximity_frames = 5  # Adjust this threshold as needed
max_divergence_threshold = 50  # Adjust this threshold as needed

df1 = filter_by_roi(df1, x_min, x_max)
df2 = filter_by_roi(df2, x_min, x_max)    

#df1 = set_time_range(df1, start_frame, end_frame)
#df2 = set_time_range(df2, start_frame, end_frame)

# Get total number of comparisons for progress bar
track_ids1 = df1['tracking_id'].unique()
track_ids2 = df2['tracking_id'].unique()
print(len(track_ids2), 'Tracks to compare:', track_ids2)

target_track = df1[df1['tracking_id'] == target_id]

plt.plot(target_track['pitch_x'], target_track['pitch_y'], 'k--', label=f'Target Track {target_id}')

for track_id in track_ids2:
    match_type, frame_diff = is_near_edge_of_track(target_track, df2[df2['tracking_id'] == track_id])
    
    proximity = False
    direction = False
    velocity = False
    duration = False
    max_divergence = False


    if match_type:
        proximity = True 
        angle = direction_analysis(target_track, df2[df2['tracking_id'] == track_id], frame_diff, match_type, 10)
        velocity_diff = velocity_analysis(target_track, df2[df2['tracking_id'] == track_id], frame_diff, match_type, 10)
        proximity_duration = check_track_proximity_duration(
            target_track, 
            df2[df2['tracking_id'] == track_id], 
            frame_diff, 
            match_type
        )
        max_distance = check_track_divergence(
            target_track,
            df2[df2['tracking_id'] == track_id],
            frame_diff,
            match_type
        )

        if angle < angle_threshold:
            print(f'Track {track_id} angle: {angle}')
            direction = True

        if velocity_diff < velocity_threshold:
            print(f'Track {track_id} velocity difference: {velocity_diff}')
            velocity = True
            
        if proximity_duration >= min_proximity_frames:
            print(f'Track {track_id} proximity duration: {proximity_duration} frames')  
            duration = True
            
        if max_distance <= max_divergence_threshold:
            print(f'Track {track_id} maximum divergence: {max_distance:.2f}')
            max_divergence = True

        if proximity and direction and velocity and duration and max_divergence:
            
            plt.plot(df2[df2['tracking_id'] == track_id]['pitch_x'], 
                    df2[df2['tracking_id'] == track_id]['pitch_y'], 
                    '--', 
                    label=f'Track {track_id}')
            


plt.legend()
plt.show()