"""run_v2.py with version 3 evidence: name questions get synonyms, a boost for chunks with a name near the entity
and 8 sessions (retrieval.py, rank_chunks()). Same flags as run_v2.py, --no-hint and --lenient included.

    python run_v3.py --questions sample_hard.jsonl --histories dev-large/histories.jsonl \\
        --base-url https://tNN.<domain> --agent-id agt_... --out results-v3.jsonl --no-hint --lenient
"""
from __future__ import annotations

import sys

from run_v2 import main

VERSION = 3

if __name__ == "__main__":
    sys.exit(main(version=VERSION))
