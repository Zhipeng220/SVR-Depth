"""Group saved continuous trajectories by actual age, without rerunning a model."""
from collections import defaultdict

def summarize_by_age(rows):
    groups = defaultdict(list)
    streams = defaultdict(list)
    for row in rows:
        if not str(row.get('age', '')).strip() or not str(row.get('is_keyframe', '')).strip():
            raise ValueError('Every frame record needs actual age and is_keyframe')
        age = int(row['age'])
        is_keyframe = str(row['is_keyframe']).strip().lower()
        if age < 0 or is_keyframe not in ('true', 'false', '1', '0'):
            raise ValueError('Invalid age or keyframe flag')
        keyframe = is_keyframe in ('true', '1')
        if keyframe != (age == 0):
            raise ValueError('Keyframe must have age zero; non-keyframes must have positive age')
        key = tuple(row[k] for k in ('dataset', 'sequence', 'method', 'seed', 'k'))
        streams[key].append((row['index'], age))
        groups[(*key, age)].append(row)
    for key, states in streams.items():
        states.sort()
        if not states or states[0] != (0, 0):
            raise ValueError(f'Trajectory must start at RGB0 keyframe: {key}')
        for (previous_index, previous_age), (index, age) in zip(states, states[1:]):
            if index != previous_index + 1 or (age != 0 and age != previous_age + 1):
                raise ValueError(f'Noncontinuous trajectory or invalid age transition: {key}, frame {index}')
    results = []
    for key, items in sorted(groups.items()):
        values = [r['absrel'] for r in items if r['status'] == 'ok']
        results.append(dict(dataset=key[0], sequence=key[1], method=key[2], seed=key[3],
            k=key[4], age=key[5], total_frames=len(items), valid_frames=len(values),
            invalid_or_unmatched=len(items)-len(values),
            mean_absrel=sum(values)/len(values) if values else None))
    return results
