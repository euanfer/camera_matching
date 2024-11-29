import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from tqdm import tqdm

def filter_by_roi(df, min_x, max_x):
    """Filter dataframe to only include points within ROI"""
    return df[(df['pitch_x'] >= min_x) & (df['pitch_x'] <= max_x)]

def find_closest_point(target_point, comparison_track):
    """Find closest point in comparison track to target track edge"""
    distances = comparison_track.apply(
        lambda row: euclidean(
            [target_point['pitch_x'], target_point['pitch_y']], 
            [row['pitch_x'], row['pitch_y']]
        ), 
        axis=1
    )
    return comparison_track.loc[distances.idxmin()]

def is_near_edge_of_track(target_track, comparison_track, distance_threshold=30, frame_threshold=90):
    """Check if track is within threshold of edge of target track"""
    target_start = target_track.iloc[0]
    target_end = target_track.iloc[-1]

    start_match = find_closest_point(target_start, comparison_track)
    end_match = find_closest_point(target_end, comparison_track)

    start_dist = euclidean([target_start['pitch_x'], target_start['pitch_y']], 
                          [start_match['pitch_x'], start_match['pitch_y']])
    end_dist = euclidean([target_end['pitch_x'], target_end['pitch_y']], 
                         [end_match['pitch_x'], end_match['pitch_y']])

    start_frame_diff = target_start['frame'] - start_match['frame']
    end_frame_diff = target_end['frame'] - end_match['frame']

    if start_dist < distance_threshold and abs(start_frame_diff) < frame_threshold:
        return 'start_match', start_frame_diff, start_dist

    if end_dist < distance_threshold and abs(end_frame_diff) < frame_threshold:
        return 'end_match', end_frame_diff, end_dist

    return None, None, None

def get_direction_vector(track, start_frame, end_frame):
    """Calculate direction vector between two points in a track"""
    start_point = track[track['frame'] == start_frame]
    end_point = track[track['frame'] == end_frame]
    
    if start_point.empty or end_point.empty:
        return None
    
    dx = end_point.iloc[0]['pitch_x'] - start_point.iloc[0]['pitch_x']
    dy = end_point.iloc[0]['pitch_y'] - start_point.iloc[0]['pitch_y']
    
    magnitude = (dx**2 + dy**2)**0.5
    if magnitude == 0:
        return None
        
    return (dx/magnitude, dy/magnitude)

def direction_analysis(target_track, comparison_track, frame_diff, match_type, direction_step=10):
    """Compare direction vectors of target and comparison track"""
    
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
        return 180
    
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
        comp_start_frame = comp_end_frame - direction_step
    
    target_vel = get_velocity(target_track, target_start_frame, target_end_frame)
    comp_vel = get_velocity(comparison_track, comp_start_frame, comp_end_frame)
    
    if target_vel is None or comp_vel is None:
        return None
    
    # Calculate velocity difference magnitude
    vel_diff = ((target_vel[0] - comp_vel[0])**2 + 
                (target_vel[1] - comp_vel[1])**2)**0.5
    
    return vel_diff

def score_match(angle, velocity_diff, proximity_dist, proximity_duration, target_team, comp_team,
                angle_weight=0.2, velocity_weight=0.2, 
                proximity_weight=0.15, duration_weight=0.35,
                team_weight=0.1):
    """Calculate match score (lower is better)"""
    angle_score = min(angle / 30.0, 1.0)  # Normalize to 0-1
    velocity_score = min(velocity_diff, 1.0)
    proximity_score = min(proximity_dist / 30.0, 1.0)
    duration_score = 1 - min(proximity_duration / 30.0, 1.0)  # Inverse because longer duration is better
    team_score = 0.0 if target_team == comp_team else 1.0  # 0 if teams match, 1 if they don't
    
    return (angle_score * angle_weight + 
            velocity_score * velocity_weight + 
            proximity_score * proximity_weight +
            duration_score * duration_weight +
            team_score * team_weight)

def check_track_proximity_duration(target_track, comparison_track, frame_diff, match_type, distance_threshold=30):
    """
    Check how long tracks stay close together starting from match point
    Returns: number of frames tracks stay within threshold distance
    """
    if match_type == 'start_match':
        # Start from beginning of tracks and move forward
        target_frames = target_track['frame'].sort_values()
    else:  # end_match
        # Start from end of tracks and move backward
        target_frames = target_track['frame'].sort_values(ascending=False)
    
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

def check_track_divergence(target_track, comparison_track, frame_diff, match_type, distance_threshold=50):
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

def plot_matches(df1, df2, matches_df):
    """Plot all matched trajectories"""
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    
    # Define a list of colors for different matches
    colors = plt.cm.tab20(np.linspace(0, 1, len(matches_df)))
    
    # Plot each matched pair
    for (_, match), color in zip(matches_df.iterrows(), colors):
        # Get source track
        source_track = df1[df1['tracking_id'] == match['sprint_id']]
        # Get matched track
        matched_track = df2[df2['tracking_id'] == match['matched_track']]
        
        # Plot source track (sprint) as solid line
        plt.plot(source_track['pitch_x'], source_track['pitch_y'], 
                '-', color=color,
                label=f"Sprint {match['sprint_id']}")
        
        # Plot matched track as dashed line in same color
        plt.plot(matched_track['pitch_x'], matched_track['pitch_y'], 
                '--', color=color,
                label=f"Match {match['matched_track']}")
        
        # Add text annotation with match score
        mid_x = source_track['pitch_x'].mean()
        mid_y = source_track['pitch_y'].mean()
        plt.annotate(f"Score: {match['score']:.2f}", 
                    (mid_x, mid_y), 
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.title('Matched Trajectories')
    plt.xlabel('pitch_x')
    plt.ylabel('pitch_y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    sprints_df = pd.read_csv('csvs/sprints_cam2_1st.csv')
    df1 = pd.read_csv('csvs/cam2_1.csv')
    df2 = pd.read_csv('csvs/cam1_1.csv')
    output_path = 'csvs/matches2_1.csv'
    # Define ROI and thresholds
    x_min, x_max = 0, 1050

    angle_threshold = 30
    velocity_threshold = 1
    min_proximity_frames = 5  # Minimum frames tracks should stay close
    max_divergence_threshold = 50  # Maximum allowed divergence
    
    df1_roi = filter_by_roi(df1, x_min, x_max)
    df2_roi = filter_by_roi(df2, x_min, x_max)
    
    matches = []
    
    # Add tqdm progress bar
    print("Processing sprints...")
    for _, sprint in tqdm(sprints_df.iterrows(), total=len(sprints_df), desc="Finding matches"):
        track_id = sprint['tracking_id']
        target_track = df1_roi[df1_roi['tracking_id'] == track_id]
        target_team = target_track.iloc[0]['team_id']  # Get team ID from target track
        
        if target_track.empty:
            continue
            
        best_match = None
        best_score = float('inf')
        
        # Check all potential matches
        for comp_id in df2_roi['tracking_id'].unique():
            comparison_track = df2_roi[df2_roi['tracking_id'] == comp_id]
            comp_team = comparison_track.iloc[0]['team_id']  # Get team ID from comparison track
            match_type, frame_diff, proximity = is_near_edge_of_track(target_track, comparison_track)
            
            if match_type:
                angle = direction_analysis(target_track, comparison_track, frame_diff, match_type)
                velocity_diff = velocity_analysis(target_track, comparison_track, frame_diff, match_type)
                proximity_duration = check_track_proximity_duration(
                    target_track, comparison_track, frame_diff, match_type
                )
                max_distance = check_track_divergence(
                    target_track, comparison_track, frame_diff, match_type
                )
                
                # Only consider as match if all criteria are met
                if (angle < angle_threshold and 
                    velocity_diff is not None and 
                    velocity_diff < velocity_threshold and
                    proximity_duration >= min_proximity_frames and
                    max_distance <= max_divergence_threshold):
                    
                    match_score = score_match(
                        angle, velocity_diff, proximity, proximity_duration,
                        target_team, comp_team
                    )
                    
                    if match_score < best_score:
                        best_score = match_score
                        best_match = {
                            'sprint_id': track_id,
                            'matched_track': comp_id,
                            'frame': sprint['start_frame'] if match_type == 'start_match' else sprint['end_frame'],
                            'frame_difference': frame_diff,
                            'match_type': match_type,
                            'angle': angle,
                            'velocity_diff': velocity_diff,
                            'proximity_duration': proximity_duration,
                            'max_distance': max_distance,
                            'target_team': target_team,
                            'matched_team': comp_team,
                            'score': match_score
                        }
        
        if best_match:
            matches.append(best_match)
    
    # Save results
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(output_path, index=False)
    print(f"Found {len(matches)} matches")
    
    # Plot the matches
    plot_matches(df1, df2, matches_df)
    
    # Optional: Save the plot
    plt.savefig('sprint_matches.png', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()
