KNOWLEDGE_ACTIONS = {
    5: {"name": "assignment_balanced_landmark", "description": "Use agent id to balance agents across landmarks."},
    6: {"name": "nearest_landmark_with_tie_break", "description": "Move toward the nearest landmark with deterministic tie breaking."},
    7: {"name": "help_uncovered_region", "description": "Move toward the landmark that appears least covered by the team."},
}

import numpy as np

BASE_N_ACTIONS = 5


def map_knowledge_action(processed_obs, agent_id, action_id, threshold=0.08):
    action_id = int(action_id)
    if action_id < BASE_N_ACTIONS:
        return action_id
    n_agents = _n_agents(processed_obs)
    obs = processed_obs[agent_id]
    local_id = _agent_index(agent_id)
    semantic_index = action_id - BASE_N_ACTIONS

    nearest_other = _nearest_other(processed_obs, agent_id)
    if semantic_index in (2, 3) and nearest_other is not None and np.linalg.norm(nearest_other) < 0.16:
        return _move_away(nearest_other, threshold)

    if semantic_index == 0:
        target = _nearest_uncovered_landmark(processed_obs, agent_id)
    elif semantic_index == 1:
        target = _landmarks(obs, n_agents)[local_id % n_agents]
    elif semantic_index == 2:
        target = _nearest_other(processed_obs, agent_id)
        return 0 if target is None else _move_away(target, threshold)
    elif semantic_index == 3:
        if _on_any_landmark(processed_obs, agent_id, threshold):
            return 0
        target = _landmarks(obs, n_agents)[local_id % n_agents]
    else:
        target = _landmarks(obs, n_agents)[semantic_index % n_agents]

    return _move_towards(target, threshold)


def _agent_index(agent_id):
    return int(str(agent_id).split("_")[-1])


def _n_agents(processed_obs):
    return len(processed_obs)


def _as_vec(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    return arr[:2]


def _landmarks(obs, n_agents):
    return [_as_vec(v) for v in list(obs)[:n_agents]]


def _others(obs, n_agents):
    return [_as_vec(v) for v in list(obs)[n_agents:]]


def _on_any_landmark(processed_obs, agent_id, threshold):
    n_agents = _n_agents(processed_obs)
    return any(np.linalg.norm(v) < threshold for v in _landmarks(processed_obs[agent_id], n_agents))


def _nearest_other(processed_obs, agent_id):
    n_agents = _n_agents(processed_obs)
    others = _others(processed_obs[agent_id], n_agents)
    if not others:
        return None
    return min(others, key=lambda v: np.linalg.norm(v))


def _nearest_uncovered_landmark(processed_obs, agent_id, cover_threshold=0.12):
    n_agents = _n_agents(processed_obs)
    landmarks = _landmarks(processed_obs[agent_id], n_agents)
    covered = set()
    for other_id, other_obs in processed_obs.items():
        if other_id == agent_id:
            continue
        for idx, rel in enumerate(_landmarks(other_obs, n_agents)):
            if np.linalg.norm(rel) < cover_threshold:
                covered.add(idx)
    candidates = [(idx, lm) for idx, lm in enumerate(landmarks) if idx not in covered]
    if not candidates:
        candidates = list(enumerate(landmarks))
    return min(candidates, key=lambda item: np.linalg.norm(item[1]))[1]


def _move_towards(relative_pos, threshold):
    dx, dy = _as_vec(relative_pos)
    if np.linalg.norm([dx, dy]) < threshold:
        return 0
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 1
    return 4 if dy > 0 else 3


def _move_away(relative_pos, threshold):
    dx, dy = _as_vec(relative_pos)
    if np.linalg.norm([dx, dy]) > max(threshold, 0.16):
        return 0
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 2
    return 3 if dy > 0 else 4
