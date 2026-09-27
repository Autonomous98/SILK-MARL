import re
import numpy as np
import importlib
import os
import glob


def convert_reward(reward_dict):
    '''
    param: reward (dict): dict containing rewards for each agent.
    return: list: list containing rewards for each agent.
    '''
    rewards = []
    for agent_id, reward in reward_dict.items():
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()
        else:
            reward = [reward]
        rewards.append(reward)
    return rewards

def dict2array(dict):
    '''
    param: dict (dict): dict containing rewards for each agent. 
    return: array: array containing rewards for each agent. (nx1)
    '''
    arr = []
    for agent_id, val in dict.items():
        arr.append(val)
    return np.array(arr).reshape(-1, 1)

def convert_actions(action_list):
    n = len(action_list)
    agent_ids = [f"agent_{i}" for i in range(n)]
    action_dict = {}
    for agent_id, action in zip(agent_ids, action_list):
        action_dict[agent_id] = action
    return action_dict


def clean_obs_code(obs_code_str):
    cleaned_code = re.sub(
        r'^(import .+|from .+)', '', obs_code_str, flags=re.MULTILINE).strip()
    return cleaned_code


def process_available_actions(available_actions):
    '''
    param:
        available_actions (list of n list): list of available actions for each agent.
    return:
        action_dict (Dict of n list): dict containing available actions indices for each agent.
    '''
    n_agents = len(available_actions)
    action_dict = {}
    for i in range(n_agents):
        indices = np.where(np.array(available_actions[i]) == 1)[0].tolist()
        action_dict[f"agent_{i}"] = indices
    return action_dict


def get_generated_code(gencode_path):
    """Load the generated planning and reward functions from a Python module."""
    with open(gencode_path, "r", encoding="utf-8") as f:
        code_str = f.read()
    namespace = {**globals(), "__file__": os.path.abspath(gencode_path)}
    exec(compile(code_str, gencode_path, "exec"), namespace)
    try:
        planning_function = namespace["planning_function"]
        compute_reward = namespace["compute_reward"]
    except KeyError as exc:
        raise ImportError(
            f"Generated code must define planning_function and compute_reward: {gencode_path}"
        ) from exc
    return planning_function, compute_reward

def get_gencode_path(env_name):
    prompt_dir = os.path.dirname(os.path.abspath(__file__))
    gencode_dir = os.path.join(prompt_dir, "gen_code", env_name, "code")

    candidates = [
        path for path in glob.glob(os.path.join(gencode_dir, "*_generated_code_*.py"))
        if os.path.isfile(path)
        and not os.path.basename(path).startswith(".")
        and re.search(r"_generated_code_(\d+)\.py$", os.path.basename(path))
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No generated code found in: {gencode_dir}"
        )

    def generation_index(path):
        filename = os.path.basename(path)
        match = re.search(r"_generated_code_(\d+)\.py$", filename)
        return int(match.group(1)) if match else -1

    return max(
        candidates,
        key=lambda path: (generation_index(path), os.path.getmtime(path))
    )


def setup_wrapper(env, gencode_path):
    planning_function, compute_reward = get_generated_code(gencode_path)
    env.set_func(planning_function, compute_reward)
    return env

def lbf_task_to_actions(task, processed_obs):
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

def get_relative_position(agent_pos, target_pos):
    '''
    Helper function to calculate the relative position of target to the agent.
    Returns a tuple (dx, dy) where:
    dx: Difference in the x-coordinate.
    dy: Difference in the y-coordinate.
    '''
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
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

def constant_reward_signal(actions, llm_actions, llm_reward=0.01, penalty=0.01):
    reward_dict = {agent: 0 for agent in actions.keys()}
    # Reward for following LLM suggestions
    for agent, llm_action in llm_actions.items():
        if actions[agent] in llm_action:
            reward_dict[agent] += llm_reward
        else:
            reward_dict[agent] -= penalty

    return reward_dict

def normalized_reward(reward, theta=0.01):
    min_reward = min(reward.values())
    max_reward = max(reward.values())
    if min_reward != max_reward:
        for agent_id in reward:
            reward[agent_id] = (reward[agent_id] - min_reward) / (max_reward - min_reward) * theta
    return reward

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

def get_rware_actions(direction, relative_pos, can_move_forward):
    '''
    Helper function to determine the movement actions based on the relative position.
    Returns a list of movement actions [move1, move2].
    1: Up (X-)
    2: Down (X+)
    3: Left (Y-)
    4: Right (Y+)
    '''
    dx, dy = relative_pos

    actions = []
    if dx == 0 and dy == 0:
        return [4]
    if direction == 'up': 
        if can_move_forward and dy < 0:
            actions.append(1)
        if dx < 0:
            actions.append(2)
        if dx > 0:
            actions.append(3)
        if dx == 0:
            if dy > 0:
                actions.extend([2, 3])
            elif dy < 0 and not can_move_forward:
                actions.extend([2, 3])

    elif direction == 'down':
        if can_move_forward and dy > 0:
            actions.append(1)
        if dx < 0:
            actions.append(3)
        if dx > 0:
            actions.append(2)
        if dx == 0:
            if dy < 0:
                actions.extend([2, 3])
            elif dy > 0 and not can_move_forward:
                actions.extend([2, 3])

    elif direction == 'left':
        if can_move_forward and dx < 0:
            actions.append(1)
        if dy < 0:
            actions.append(3)
        if dy > 0:
            actions.append(2)
        if dy == 0:
            if dx > 0:
                actions.extend([2, 3])
            elif dx < 0 and not can_move_forward:
                actions.extend([2, 3])
        
    elif direction == 'right':
        if can_move_forward and dx > 0:
            actions.append(1)
        if dy < 0:
            actions.append(2)
        if dy > 0:
            actions.append(3)
        if dy == 0:
            if dx < 0:
                actions.extend([2, 3])
            elif dx > 0 and not can_move_forward:
                actions.extend([2, 3])

    return actions

def rware_task_to_actions(task, processed_obs):
    action = {agent: [] for agent in task.keys()}
    agent_infos = processed_obs['agent_infos']

    for agent in task.keys():
        agent_task = task[agent]
        agent_index = int(agent.split('_')[1])  # Extract the index from the agent string
        agent_info = agent_infos[agent_index]
        agent_pos = agent_info['location']
        agent_direction = agent_info['direction']
        agent_can_move_forward = agent_info['can_move_forward']
        agent_can_place_shelf = agent_info['can_place_shelf']

        if agent_task == "random explore":
            if agent_info['can_move_forward']:
                action[agent].extend([1, 2, 3])
            else:
                action[agent].extend([2, 3])

        elif agent_task == "empty shelf":
            empty_shelf_pos = processed_obs['empty_shelves_pos']
            if not empty_shelf_pos:
                if agent_info['can_move_forward']:
                    action[agent].extend([1, 2, 3])
                else:
                    action[agent].extend([2, 3])
            else:
                target = min(empty_shelf_pos, key=lambda pos: sum(abs(a - b) for a, b in zip(pos, agent_pos)))
                relative_pos = get_relative_position(agent_pos, target)
                action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
        
        elif agent_task == "workstation":
            workstation_locations = processed_obs['workstation location']
            closest_workstation = min(workstation_locations, 
                    key=lambda pos: sum(abs(a - b) for a, b in zip(pos, agent_pos)))
            relative_pos = get_relative_position(agent_pos, closest_workstation)
            action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
            # workstation_pos_1 = processed_obs['workstation location'][0]
            # workstation_pos_2 = processed_obs['workstation location'][1]
            # relative_pos_1 = get_relative_position(agent_pos, workstation_pos_1)
            # relative_pos_2 = get_relative_position(agent_pos, workstation_pos_2)
            # action[agent].extend(get_rware_actions(agent_direction, relative_pos_1, agent_can_move_forward))
            # action[agent].extend(get_rware_actions(agent_direction, relative_pos_2, agent_can_move_forward))

        elif agent_task == "return":
            return_locations = processed_obs['return location']
            if not processed_obs['return location']:
                if agent_info['can_move_forward']:
                    action[agent].extend([1, 2, 3])
                else:
                    action[agent].extend([2, 3])
            else:
                closest_return = min(return_locations, 
                        key=lambda pos: sum(abs(a - b) for a, b in zip(pos, agent_pos)))
                relative_pos = get_relative_position(agent_pos, closest_return)
                action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
                # for return_pos in processed_obs['return location']: 
                #     relative_pos = get_relative_position(agent_pos, return_pos)
                #     action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
        
    return action

if __name__ == "__main__":
    print(get_gencode_path("mpe_simple_spread_v3_5"))
