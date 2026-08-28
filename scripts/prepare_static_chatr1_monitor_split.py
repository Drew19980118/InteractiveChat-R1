#!/usr/bin/env python3
"""Create the conversation-disjoint monitor split for static ChatR1 data.

The implementation is shared with the static ConvAgent baseline because both
released Parquet formats encode a full static conversation in their prompt.
Keeping this small named entry point makes the ChatR1 training recipe
self-contained without duplicating split logic.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("prepare_static_convagent_monitor_split.py")),
        run_name="__main__",
    )
