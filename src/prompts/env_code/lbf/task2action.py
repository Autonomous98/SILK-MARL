import importlib
import re

def task_to_actions(task, processed_obs):
    '''
    param: task (dict): task to be converted to actions, keyed by agent_id (e.g. 'agent_0').
    param: processed_obs: tuple (food_info, other_agents_info) containing information about food and other agents.
    return: dict: dictionary of actions for each agent.
    '''
    food_info, agents_info = processed_obs
    tasks = task if isinstance(task, dict) else {}
    actions = {agent: [0] for agent in agents_info}

    for agent, agent_task in tasks.items():
        if agent not in agents_info or not isinstance(agent_task, str):
            continue

        task_name = agent_task.strip()
        if task_name == "No op":
            actions[agent] = [0]
            continue
        if task_name == "Pickup":
            actions[agent] = [5]
            continue

        match = re.fullmatch(r"Target food (\d+)", task_name)
        if match is None:
            continue

        food = food_info.get(f"food_{int(match.group(1))}")
        if food is None:
            continue

        agent_pos = agents_info[agent][0]
        food_pos = food[0]
        relative_pos = get_relative_position(agent_pos, food_pos)
        actions[agent] = get_movement_actions(relative_pos) or [0]

    return actions

def get_relative_position(agent_pos, food_pos):
    '''
    Helper function to calculate the relative position of food to the agent.
    Returns a tuple (dx, dy) where:
    dx: Difference in the x-coordinate.
    dy: Difference in the y-coordinate.
    '''
    dx = food_pos[0] - agent_pos[0]
    dy = food_pos[1] - agent_pos[1]
    return dx, dy

def get_movement_actions(relative_pos):
    '''
    Helper function to determine the movement actions based on the relative position.
    Returns a list of movement actions [move1, move2].
    1: Move North (X-)
    2: Move South (X+)
    3: Move West (Y-)
    4: Move East (Y+)
    '''
    dx, dy = relative_pos
    actions = []
    
    # Determine movement in the x-direction
    if dx < 0:
        actions.append(1)  # Move North
    elif dx > 0:
        actions.append(2)  # Move South
    
    # Determine movement in the y-direction
    if dy < 0:
        actions.append(3)  # Move West
    elif dy > 0:
        actions.append(4)  # Move East
    
    return actions


def import_function(module_name, func_name):
    try:
        # Attempt to import the module
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return func
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        return None
