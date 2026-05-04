import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="agent-forge - Multi-turn agent orchestration with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config.yaml
  python main.py --config config.yaml --debug
  python main.py --config config.yaml --log-file session.log
        """,
    )

    # Required arguments
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")

    # Optional flags
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")

    parser.add_argument("--log-file", help="Path to log file for session logging")

    parser.add_argument(
        "--version", action="version", version="%(prog)s 1.0.0", help="Show version information and exit"
    )

    return parser.parse_args()
