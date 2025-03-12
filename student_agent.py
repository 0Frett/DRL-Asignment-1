import numpy as np
import pickle
import random
import gym

# Global variables to track our agent's internal state across calls.
# (Note: In a real training setup you might prefer a learned Q-value table or network,
#  but here we use a simple rule-based heuristic for clarity.)
has_passenger = False      # True after a successful pickup
pickup_station = None      # The station where the passenger was picked up
target = None              # Current target station (either to search for a passenger or deliver)
stations_tried = []        # For search: stations already visited (and found no passenger)

def plan_move(current, target, obstacles):
    """
    Given the current position and a target position,
    choose a movement action (0:South, 1:North, 2:East, 3:West)
    while checking if the desired direction is blocked.
    """
    cur_row, cur_col = current
    tar_row, tar_col = target
    
    # Try vertical movement first
    if cur_row < tar_row and obstacles.get("south", 0) == 0:
        return 0  # Move South
    if cur_row > tar_row and obstacles.get("north", 0) == 0:
        return 1  # Move North
    # Then try horizontal movement
    if cur_col < tar_col and obstacles.get("east", 0) == 0:
        return 2  # Move East
    if cur_col > tar_col and obstacles.get("west", 0) == 0:
        return 3  # Move West

    # If the desired direction is blocked, try any allowed move (prioritizing a default order)
    for move in [0, 1, 2, 3]:
        if move == 0 and obstacles.get("south", 0) == 0:
            return 0
        if move == 1 and obstacles.get("north", 0) == 0:
            return 1
        if move == 2 and obstacles.get("east", 0) == 0:
            return 2
        if move == 3 and obstacles.get("west", 0) == 0:
            return 3
    # If no movement is available (should be rare), choose randomly among movement actions.
    return random.choice([0, 1, 2, 3])

def get_action(obs):
    """
    Decide which action to take given the observation.
    
    Observation format:
      (taxi_row, taxi_col, station1_row, station1_col, station2_row, station2_col,
       station3_row, station3_col, station4_row, station4_col,
       obstacle_north, obstacle_south, obstacle_east, obstacle_west,
       passenger_look, destination_look)
    
    Actions:
      0: Move South
      1: Move North
      2: Move East
      3: Move West
      4: PICKUP
      5: DROPOFF
    """
    global has_passenger, pickup_station, target, stations_tried

    taxi_row, taxi_col = obs[0], obs[1]
    # The four stations (R, G, Y, B) are fixed (given in the observation)
    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
    # Obstacle flags: 1 means blocked in that direction.
    obstacles = {
        "north": obs[10],
        "south": obs[11],
        "east": obs[12],
        "west": obs[13]
    }
    passenger_look = obs[14]
    destination_look = obs[15]
    current_pos = (taxi_row, taxi_col)

    # ----- Phase 1: Searching for the Passenger -----
    if not has_passenger:
        # If we are at a station, check if the passenger is here.
        for st in stations:
            if current_pos == st:
                if passenger_look:
                    # Passenger found at this station: pick up!
                    has_passenger = True
                    pickup_station = st
                    # Reset target and tried stations for the next phase.
                    target = None
                    stations_tried = []
                    return 4  # Execute PICKUP
                else:
                    # Mark this station as visited (passenger not found here)
                    if st not in stations_tried:
                        stations_tried.append(st)
        # If no current target or if we’ve reached our target, choose the nearest unvisited station.
        if target is None or current_pos == target:
            remaining = [st for st in stations if st not in stations_tried]
            if not remaining:
                remaining = stations  # If all have been visited, reset (to cycle through stations)
            target = min(remaining, key=lambda st: abs(taxi_row - st[0]) + abs(taxi_col - st[1]))
        # Move toward the chosen target station.
        return plan_move(current_pos, target, obstacles)

    # ----- Phase 2: Delivering the Passenger -----
    else:
        # The destination is one of the stations other than where we picked up the passenger.
        candidate_dest = [st for st in stations if st != pickup_station]
        # If we are at a candidate destination and the destination is in view, execute dropoff.
        for st in candidate_dest:
            if current_pos == st and destination_look:
                # Reset state for the next episode (episode will end after a successful dropoff)
                has_passenger = False
                pickup_station = None
                target = None
                stations_tried = []
                return 5  # Execute DROPOFF
        # If no target is set or we have reached our target, choose the nearest candidate destination.
        if target is None or current_pos == target:
            target = min(candidate_dest, key=lambda st: abs(taxi_row - st[0]) + abs(taxi_col - st[1]))
        # Move toward the chosen destination.
        return plan_move(current_pos, target, obstacles)
