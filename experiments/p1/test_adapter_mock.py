
import torch

from stream_state import StreamState
from stream_adapter import StreamAdapter, StepResult


# 模拟普通传播
def fake_propagate(frame, state):
    state.radius_teacher = state.radius_teacher + frame
    return StepResult(
        depth=state.radius_teacher.clone(),
        state=state,
    )


# 模拟刷新
def fake_refresh(frame, state):
    state.radius_teacher = state.radius_teacher + 10
    return StepResult(
        depth=state.radius_teacher.clone(),
        state=state,
    )


# 初始化状态
initial = StreamState(
    radius_teacher=torch.tensor([1.0])
)

adapter = StreamAdapter(
    propagate_fn=fake_propagate,
    refresh_fn=fake_refresh,
)

result = adapter.branch(
    frame=torch.tensor([2.0]),
    incoming_state=initial,
)

# B0：传播输出与状态 = 3
assert result.b0.depth.item() == 3.0
assert result.b0.state.radius_teacher.item() == 3.0

# B1：刷新输出与状态 = 13
assert result.b1.depth.item() == 13.0
assert result.b1.state.radius_teacher.item() == 13.0

# B2：当前输出 = 13，持久状态 = 3
assert result.b2.depth.item() == 13.0
assert result.b2.state.radius_teacher.item() == 3.0

# B1/B2 当前输出完全一致
assert torch.equal(result.b1.depth, result.b2.depth)

# 检查分支独立性
result.b1.state.radius_teacher.fill_(99)

assert result.b2.state.radius_teacher.item() == 3.0
assert initial.radius_teacher.item() == 1.0

print("PASS: B0/B1/B2 动作语义测试")
print("PASS: B1/B2 当前输出一致")
print("PASS: B1/B2 持久状态不同")
print("PASS: 分支状态相互隔离")
