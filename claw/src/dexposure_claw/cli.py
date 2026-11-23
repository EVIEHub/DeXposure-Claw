"""Command-line interface for DeXposure Claw."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .installer import build_claude_code_plugin, install, render_install_snippet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dexposure-claw")
    sub = parser.add_subparsers(dest="command")

    install_parser = sub.add_parser("install", help="Install into an agent runtime")
    install_parser.add_argument(
        "target",
        nargs="?",
        choices=["claude-code", "hermes", "codex", "mcp"],
        help="Runtime target. Omit to print available options.",
    )
    install_parser.add_argument("--output-dir", type=Path)

    build = sub.add_parser("build", help="Build runtime adapter artifacts")
    build.add_argument("target", choices=["claude-code"])
    build.add_argument("--output-dir", type=Path)

    sub.add_parser("mcp", help="Start the MCP server")
    sub.add_parser("health", help="Print package health information")

    snippet = sub.add_parser("snippet", help="Print manual install config")
    snippet.add_argument("target", choices=["hermes", "codex", "mcp"])

    live = sub.add_parser(
        "live-predict",
        help="Collect live DeFiLlama exposure network, run DeXposure-FM, and output Figure 2 analysis",
    )
    live.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top TVL protocols to ingest from DeFiLlama (default: 30)",
    )
    live.add_argument(
        "--horizon",
        type=int,
        choices=[1, 4, 8, 12],
        default=4,
        help="Forecast horizon in weeks (default: 4)",
    )
    live.add_argument(
        "--hub-id",
        type=str,
        default="2786",
        help="Focal hub protocol/asset ID for Figure 2 comparison (default: 2786)",
    )
    live.add_argument(
        "--min-node-size",
        type=float,
        default=100_000.0,
        help="Minimum node TVL in USD (default: 100000.0; matches DeXposure pruning threshold)",
    )
    live.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to save the JSON output",
    )

    serve_cmd = sub.add_parser(
        "serve",
        help="Deploy the FastAPI live prediction & FM server",
    )
    serve_cmd.add_argument("--host", type=str, default="0.0.0.0")
    serve_cmd.add_argument("--port", type=int, default=8000)

    return parser


def _ensure_paper_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    paper_dir = repo_root / "paper"
    if paper_dir.is_dir() and str(paper_dir) not in sys.path:
        sys.path.insert(0, str(paper_dir))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        print(json.dumps(build_claude_code_plugin(args.output_dir), indent=2))
        return 0
    if args.command == "install":
        print(json.dumps(install(args.target, args.output_dir), indent=2))
        return 0
    if args.command == "snippet":
        print(render_install_snippet(args.target))
        return 0
    if args.command == "health":
        print(json.dumps({"status": "ok", "package": "dexposure-claw"}, indent=2))
        return 0
    if args.command == "mcp":
        from .mcp_server import main as mcp_main

        return mcp_main()
    if args.command == "live-predict":
        _ensure_paper_path()
        from dexposure_agent.live_defillama import run_live_prediction_pipeline

        result = run_live_prediction_pipeline(
            top_n_protocols=args.top_n,
            horizon=args.horizon,
            hub_id=args.hub_id,
            min_node_size=args.min_node_size,
        )
        rendered = json.dumps(result, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    if args.command == "serve":
        _ensure_paper_path()
        import uvicorn

        uvicorn.run("dexposure_agent.serve:app", host=args.host, port=args.port)
        return 0

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
