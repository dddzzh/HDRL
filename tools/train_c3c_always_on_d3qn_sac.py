from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_c3c_d3qn_sac import main


if __name__ == "__main__":
    main(
        default_config="configs/hierarchical_d3qn_sac_c3c_always_on.json"
    )
