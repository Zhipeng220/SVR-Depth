#!/usr/bin/env python3
import traceback
import y10_logic as logic
from y10_core import main
if __name__=='__main__':
    try:raise SystemExit(main(logic))
    except KeyboardInterrupt:print('Interrupted; resume the printed output directory.');raise SystemExit(130)
    except Exception:traceback.print_exc();raise SystemExit(1)
