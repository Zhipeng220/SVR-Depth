#!/usr/bin/env python3
"""Read-only Y05-H4 monitor. Standard library only; no training/GPU imports."""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ARMS = ('C_current', 'F_future', 'FH_future_harm', 'P_closed_loop')
BEIJING = ZoneInfo('Asia/Shanghai')
MARKERS = re.compile(r'^\[(TRAIN [^\]]+|RESUME [^\]]+|QUALITY|PROBE COMPLETE|TIMING|CACHE [^\]]+|GPU QUEUE|PREFLIGHT PASS|VERIFIED)\].*$', re.M)


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def journal_rows(path):
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get('step'), int):
                result.append(row)
        except ValueError:
            pass
    return result


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours}小时{minutes:02d}分{seconds:02d}秒'


def show_eta(seconds):
    finish = datetime.now(BEIJING) + timedelta(seconds=seconds)
    print('当前阶段预计剩余：', duration(seconds))
    print('预计完成时间（北京时间）：', finish.strftime('%m-%d %H:%M:%S'))


def remaining_report(directory, settings, stage, phase):
    """Completion records are indicators; the experiment verifies full hashes."""
    try:
        signature = hashlib.sha256((directory / 'protocol.json').read_bytes()).hexdigest()
    except OSError:
        signature = None

    def record(path):
        value = read_json(path)
        return value if signature and value.get('signature') == signature else {}

    lines = ['\n剩余实验与后续任务（当前运行类型）：']
    pending = []
    done_count = 0
    fixed = int(settings.get('fixed_updates', 2 if phase == 'smoke' else 1000))
    for arm in ARMS:
        folder = directory / 'training' / arm
        summary = record(folder / 'summary.json')
        needed = ('equal_updates', 'equal_compute') if arm == 'P_closed_loop' else ('final',)
        done = all(name in summary.get('selected', {}) and (folder / (name + '.pt')).is_file() for name in needed)
        if done:
            done_count += 1
            detail = f"已完成，{summary.get('steps', '?')} 次更新"
        else:
            pending.append(arm)
            active = stage in ('TRAIN ' + arm, 'RESUME ' + arm)
            rows = journal_rows(folder / 'physical_updates.jsonl')
            step = rows[-1]['step'] if rows else 0
            detail = '当前阶段' if active else ('已有进度记录' if rows or (folder / 'recovery.pt').exists() else '未开始')
            if rows:
                detail += f'，日志更新 {step}'
            if arm == 'P_closed_loop':
                detail += f'；需满足 {fixed} 次更新及 FH 计算预算'
            else:
                detail += f'；目标 {fixed} 次更新'
        lines.append(f'  {arm}：{detail}')
    lines.append(f'训练组已完成：{done_count}/4；尚未完成：{4 - done_count} 组')

    checks = (
        ('全序列质量评估', 'quality/*/*.json', 51, 'csv_sha256'),
        ('未来效用探针', 'probes/*/complete.json', 6 if phase == 'smoke' else 54, 'rows_sha256'),
        ('独立计时', 'timing/*.json', 36 if phase == 'smoke' else 108, 'endpoint_verified'),
    )
    for label, pattern, total, key in checks:
        count = sum(bool(record(p).get(key)) for p in directory.glob(pattern))
        left = max(0, total - count)
        lines.append(f'  {label}：已完成 {count}/{total}，剩余 {left} 个单元')
        if left:
            pending.append(f'{label}（{left} 个单元）')

    summary = record(directory / 'summary.json')
    expected = 'SMOKE_PASS' if phase == 'smoke' else 'Y05_DEVELOPMENT_SUITE_COMPLETE'
    complete = summary.get('status') == expected and (directory / 'y05_review.zip').is_file()
    lines.append('  结果汇总与打包：' + ('已完成' if complete else '未完成'))
    if not complete:
        pending.append('结果汇总与打包')
    lines.append('仍需完成：' + (' → '.join(pending) if pending else '当前阶段全部完成'))
    if phase != 'validation':
        lines.append('注意：以上为 ' + phase + ' 状态；正式 Y05 的四组训练和后续评估仍需完成。')
    return lines, complete


def activity(run):
    paths = [run] + [run / (phase + '.log') for phase in ('validation', 'smoke', 'preflight')]
    paths += [run / phase / 'protocol.json' for phase in ('validation', 'smoke', 'preflight')]
    return max((p.stat().st_mtime_ns for p in paths if p.exists()), default=0)


def monitor(run):
    phase = next((p for p in ('validation', 'smoke', 'preflight') if (run / (p + '.log')).exists() or (run / p / 'protocol.json').exists()), 'preflight')
    directory = run / phase
    settings = read_json(directory / 'protocol.json').get('settings', {})
    print('运行目录：', run)
    print('运行类型：', {'validation': '正式实验', 'smoke': 'Smoke 检查', 'preflight': '预检查'}[phase])
    print('刷新时间（北京时间）：', datetime.now(BEIJING).strftime('%m-%d %H:%M:%S'))
    log = run / (phase + '.log')
    text = log.read_text(errors='replace') if log.is_file() else ''
    markers = list(MARKERS.finditer(text))
    latest = markers[-1] if markers else None
    stage = latest.group(1) if latest else ''
    print('\n最新进度：', latest.group(0) if latest else '等待日志更新')
    lines, complete = remaining_report(directory, settings, stage, phase)
    print('\n'.join(lines))
    if phase == 'validation' and complete:
        print('\nY05 全流程已完成：', directory / 'y05_review.zip')
        return True

    print('\n当前阶段时间估算：')
    if stage.startswith('TRAIN '):
        arm = stage.removeprefix('TRAIN ')
        journal = directory / 'training' / arm / 'physical_updates.jsonl'
        rows = journal_rows(journal)
        if rows:
            # Remove abandoned later updates when replay starts from a checkpoint.
            retained = {}
            for row in rows:
                step = row['step']
                if retained and step <= max(retained):
                    retained = {k: v for k, v in retained.items() if k < step}
                retained[step] = row
            step = rows[-1]['step']
            recent = list(retained.values())[-30:]
            wall = statistics.mean(r['step_wall_sec'] for r in recent)
            cuda = statistics.mean(r['cuda_sec'] for r in recent)
            remaining = max(0, int(settings.get('fixed_updates', 2 if phase == 'smoke' else 1000)) - step)
            if arm == 'P_closed_loop':
                fh = read_json(directory / 'training/FH_future_harm/summary.json')
                budget = fh.get('selected', {}).get('final', {}).get('cuda_sec')
                if budget is not None and cuda > 0:
                    used = sum(r['cuda_sec'] for r in retained.values())
                    remaining = max(remaining, math.ceil(max(0, budget - used) / cuda))
                    print(f'CUDA 预算：{used:.1f} / {budget:.1f} 秒')
            print(f'训练组：{arm}；日志更新：{step}；最近每步平均：{wall:.2f} 秒')
            print('日志累计物理步骤耗时：', duration(sum(r['step_wall_sec'] for r in rows)))
            if time.time() - journal.stat().st_mtime < 300:
                show_eta(remaining * wall)
            else:
                print('训练记录已超过 5 分钟未更新，暂不估算。')
        else:
            print('等待有效训练记录。')
    elif stage in ('QUALITY', 'PROBE COMPLETE', 'TIMING'):
        pattern, total = {
            'QUALITY': ('quality/*/*.json', 51),
            'PROBE COMPLETE': ('probes/*/complete.json', 6 if phase == 'smoke' else 54),
            'TIMING': ('timing/*.json', 36 if phase == 'smoke' else 108),
        }[stage]
        files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
        if len(files) >= 2 and time.time() - files[-1].stat().st_mtime < 300:
            stamps = [p.stat().st_mtime for p in files[-9:]]
            interval = statistics.median(b - a for a, b in zip(stamps, stamps[1:]))
            show_eta(max(0, total - len(files)) * interval)
            print('按最近完成单元的间隔粗估。')
        else:
            print('等待更多近期完成单元后估算。')
    else:
        print('当前处于准备、恢复或检查阶段，暂不估算。')
    if log.exists():
        print(f'日志距今更新：{time.time() - log.stat().st_mtime:.0f} 秒')
    print('日志更新可能超过最后已保存的断点；监控不代替完整性校验。')
    print('时间估算只针对当前阶段；Ctrl+C 仅退出监控。')
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Y05 根目录或其 validation/smoke 目录')
    parser.add_argument('--once', action='store_true', help='只显示一次')
    args = parser.parse_args()
    repo = next((p for p in (Path.cwd(), Path('/root/autodl-tmp/SVR-Depth/velodepth'), Path.home() / 'SVR-Depth/velodepth') if (p / 'outputs').is_dir()), Path.cwd())
    run = args.run.expanduser().resolve() if args.run else None
    if run and run.name in ('validation', 'smoke', 'preflight'):
        run = run.parent
    try:
        while True:
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            if run is None:
                runs = [p for p in (repo / 'outputs').glob('gt4a2_y05_*') if p.is_dir()]
                if runs:
                    run = max(runs, key=activity)
            if run is None:
                print('尚未找到 Y05 输出目录。')
                done = False
            else:
                done = monitor(run)
            if done or args.once:
                break
            time.sleep(15)
    except KeyboardInterrupt:
        print('\n已退出监控，训练不受影响。')


if __name__ == '__main__':
    main()
