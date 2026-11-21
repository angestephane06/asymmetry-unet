"""python -m asymunet {data,train,export,all}"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="asymunet")
    parser.add_argument("command", choices=["data", "train", "export", "all"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/default"))
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    if args.command in ("data", "all"):
        from asymunet.dataset import build_all

        build_all(args.data, args.n_train, args.n_val, args.n_test)
    if args.command in ("train", "all"):
        from asymunet.train import train

        results = train(args.data, args.out, epochs=args.epochs)
        print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))
    if args.command in ("export", "all"):
        from asymunet.export import export_and_quantize

        print(json.dumps(export_and_quantize(args.data, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
