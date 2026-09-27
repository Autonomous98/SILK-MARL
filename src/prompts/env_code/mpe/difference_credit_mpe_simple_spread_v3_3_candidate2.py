import numpy as np

STEP_SIZE = 0.1
COLLISION_WEIGHT = 0.8
ASSIGNMENT_WEIGHT = 0.5


def difference_credit_fn(prev_processed_obs, primitive_actions):
    agent_ids = sorted(prev_processed_obs.keys(), key=lambda x: int(str(x).split("_")[-1]))
    if isinstance(primitive_actions, dict):
        actions = [int(primitive_actions.get(agent_id, 0)) for agent_id in agent_ids]
    else:
        actions = [int(a) for a in primitive_actions]
    actual_score = _score(prev_processed_obs, actions)
    credits = {}
    for idx, agent_id in enumerate(agent_ids):
        counterfactual = list(actions)
        counterfactual[idx] = 0
        credits[agent_id] = float(actual_score - _score(prev_processed_obs, counterfactual))
    return credits


def _score(processed_obs, actions):
    agent_ids = sorted(processed_obs.keys(), key=lambda x: int(str(x).split("_")[-1]))
    n_agents = len(agent_ids)
    before = []
    after = []
    after_others = []
    assignment_progress = 0.0
    for idx, agent_id in enumerate(agent_ids):
        obs = list(processed_obs[agent_id])
        landmarks = [_as_vec(v) for v in obs[:n_agents]]
        others = [_as_vec(v) for v in obs[n_agents:]]
        delta = _action_delta(actions[idx] if idx < len(actions) else 0)
        before.append(landmarks)
        moved_landmarks = [lm - delta for lm in landmarks]
        after.append(moved_landmarks)
        after_others.append([other - delta for other in others])
        assigned = idx % max(1, n_agents)
        assignment_progress += np.linalg.norm(landmarks[assigned]) - np.linalg.norm(moved_landmarks[assigned])
    coverage_progress = _coverage_distance(before) - _coverage_distance(after)
    collision_penalty = _collision_penalty(after_others)
    return float(coverage_progress + ASSIGNMENT_WEIGHT * assignment_progress - COLLISION_WEIGHT * collision_penalty)


def _as_vec(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    return arr[:2]


def _coverage_distance(all_landmarks):
    n_agents = len(all_landmarks)
    total = 0.0
    for landmark_idx in range(n_agents):
        total += min(np.linalg.norm(all_landmarks[agent_idx][landmark_idx]) for agent_idx in range(n_agents))
    return total


def _collision_penalty(all_other_rel, threshold=0.12):
    penalty = 0.0
    for others in all_other_rel:
        for rel in others:
            dist = np.linalg.norm(rel)
            if dist < threshold:
                penalty += threshold - dist
    return penalty / 2.0


def _action_delta(action):
    if action == 1:
        return np.array([-STEP_SIZE, 0.0], dtype=np.float32)
    if action == 2:
        return np.array([STEP_SIZE, 0.0], dtype=np.float32)
    if action == 3:
        return np.array([0.0, -STEP_SIZE], dtype=np.float32)
    if action == 4:
        return np.array([0.0, STEP_SIZE], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)
