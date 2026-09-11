"""Put every scripts/<group>/ directory on sys.path.

The scripts are grouped into subfolders (core, data, s_backbone, drift, mos, rerank,
generators, analysis), but several of them import each other by bare module name --
`import drift_v4`, `import mf_probe`, `import coherence_dimensions` -- and those
siblings now live in a different subfolder. Importing this module makes all of them
resolvable regardless of which group the running script sits in.

Scripts that need it carry a four-line shim near the top; the shim tolerates this
module being absent, which is what keeps the flat GPU-box layout (all scripts at the
repo root, no subfolders) working unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent

for _d in [_SCRIPTS, *sorted(p for p in _SCRIPTS.iterdir()
                             if p.is_dir() and not p.name.startswith((".", "_")))]:
    _s = str(_d)
    if _s not in sys.path:
        sys.path.append(_s)
