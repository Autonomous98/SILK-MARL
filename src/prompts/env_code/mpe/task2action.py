def mpe_task_to_actions(task, processed_obs, N=None):
    """Map Landmark_<index> tasks to primitive actions for Simple Spread."""
    from collections.abc import Mapping
    import re
    import numpy as np

    tasks = task if isinstance(task, Mapping) else {}
    observations = processed_obs if isinstance(processed_obs, Mapping) else {}
    agents = list(observations)
    agents.extend(agent for agent in tasks if agent not in observations)
    actions = {agent: [0] for agent in agents}
    for agent in agents:
        assigned_task = tasks.get(agent)
        if not isinstance(assigned_task, str):
            continue
        match = re.fullmatch(r"Landmark_(\d+)", assigned_task.strip())
        if match is None:
            continue
        try:
            agent_obs = observations[agent]
            # The trailing vectors describe the other agents, not landmarks.
            landmark_count = len(agent_obs) - max(0, len(observations) - 1)
            if N is not None:
                landmark_count = min(landmark_count, int(N))
            landmark_index = int(match.group(1))
            if not 0 <= landmark_index < landmark_count:
                continue
            vector = np.asarray(agent_obs[landmark_index], dtype=float)
            if vector.shape != (2,) or not np.isfinite(vector).all():
                continue
            actions[agent] = get_mpe_actions(vector) or [0]
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            continue
    return actions

def get_mpe_actions(relative_pos):
    '''
    Helper function to determine the movement actions based on the relative position.
    Returns a list of movement actions [move1, move2].
    1: Move left x-
    2: Move right x+
    3: Move up y+
    4: Move down y-
    '''
    dx, dy = relative_pos
    actions = []
    
    # Determine movement in the x-direction
    if dx < 0:
        actions.append(1)  # Move left
    elif dx > 0:
        actions.append(2)  # Move right
    
    # Determine movement in the y-direction
    if dy < 0:
        actions.append(3)  # Move down
    elif dy > 0:
        actions.append(4)  # Move up
    
    return actions
