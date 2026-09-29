import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.offline_experiments.common import run_command


DEFAULT_RATE_SETS = [
    "1,6,12,18",
    "1,8,16,24",
    "1,12,24,36",
]
DEFAULT_CHANNELS = [32, 64, 96, 128]


def build_delegate_command(
    rates,
    channels,
    base_variant="b32_final",
    device=None,
    summary_prefix="b32_aspp_search",
    include_default=False,
    extra_args=None,
):
    command = [
        sys.executable,
        "scripts/offline_experiments/run_ablation.py",
        "--aspp_search_base",
        base_variant,
        "--aspp_rates",
        *[str(rate_set) for rate_set in rates],
        "--aspp_out_channels",
        *[str(channel) for channel in channels],
        "--aspp_summary_prefix",
        str(summary_prefix),
    ]
    if device:
        command.extend(["--device", str(device)])
    if include_default:
        command.append("--aspp_search_include_default")
    if extra_args:
        command.extend(list(extra_args))
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Search ASPP rates and ASPP output channels for HrSegNet-B32 on the custom dataset."
    )
    parser.add_argument(
        "--rates",
        nargs="*",
        default=list(DEFAULT_RATE_SETS),
        help="ASPP rate sets, each item is comma-separated. Example: 1,6,12,18 1,8,16,24",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        type=int,
        default=list(DEFAULT_CHANNELS),
        help="ASPP output channel candidates. Example: 32 64 96 128",
    )
    parser.add_argument(
        "--base_variant",
        default="b32_final",
        choices=["b32_aspp", "b32_final"],
        help="Which B32 ASPP-capable ablation config to clone for the search.",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "gpu"], help="Optional shortcut forwarded to run_ablation.py.")
    parser.add_argument("--summary_prefix", default="b32_aspp_search", help="Summary table prefix under output/offline_experiments/tables.")
    parser.add_argument("--include_default", action="store_true", help="Keep the base config in the executed experiment set.")
    parser.add_argument(
        "--log_path",
        default="output/offline_experiments/b32_aspp_search/run_b32_aspp_search.log",
        help="Wrapper log path.",
    )
    args, extra_args = parser.parse_known_args(argv)
    args.extra_args = extra_args
    return args


def main(argv=None):
    args = parse_args(argv)
    command = build_delegate_command(
        rates=args.rates,
        channels=args.channels,
        base_variant=args.base_variant,
        device=args.device,
        summary_prefix=args.summary_prefix,
        include_default=args.include_default,
        extra_args=args.extra_args,
    )
    run_command(command, args.log_path, Path.cwd())


if __name__ == "__main__":
    main()
