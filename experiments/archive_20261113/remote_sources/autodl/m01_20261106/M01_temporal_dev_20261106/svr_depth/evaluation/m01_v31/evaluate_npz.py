"""Run the provisional offline pair diagnostic on explicitly prepared NPZ arrays."""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .metrics import evaluate_pair
except ImportError:
    from metrics import evaluate_pair


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="NPZ with arrays specified in README")
    p.add_argument("--pair-id", required=True)
    p.add_argument("--flow-source", required=True)
    p.add_argument("--pose-source", required=True)
    p.add_argument("--fb-max-error-px", type=float, required=True)
    p.add_argument("--prev-pose-verified", action="store_true")
    p.add_argument("--curr-pose-verified", action="store_true")
    args = p.parse_args(argv)
    with np.load(Path(args.input), allow_pickle=False) as data:
        names = ("prev_pred", "curr_pred", "prev_gt", "curr_gt", "prev_gt_mask", "curr_gt_mask",
                 "K_prev", "K_curr", "flow_bwd_px", "flow_fwd_px", "flow_bwd_valid", "flow_fwd_valid")
        values = {name: data[name] for name in names}
        values["prev_c2w"] = data["prev_c2w"] if "prev_c2w" in data else None
        values["curr_c2w"] = data["curr_c2w"] if "curr_c2w" in data else None
    result = evaluate_pair(**values, prev_pose_valid=args.prev_pose_verified,
                           curr_pose_valid=args.curr_pose_verified,
                           fb_max_error_px=args.fb_max_error_px)
    print(json.dumps({"status": "PROVISIONAL_NOT_PAPER_VERIFIED", "pair_id": args.pair_id,
                      "flow_source": args.flow_source, "pose_source": args.pose_source,
                      "fb_max_error_px": args.fb_max_error_px, "scores": result},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
