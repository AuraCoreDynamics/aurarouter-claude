"""Entry point for ``python -m aurarouter_claude``."""

from __future__ import annotations

import sys


def main() -> None:
    """Create and run the MCP server."""
    from aurarouter_claude.server import create_server

    server = create_server()
    server.run()


if __name__ == "__main__":
    # Support --help flag for basic discoverability
    if "--help" in sys.argv or "-h" in sys.argv:
        print("aurarouter-claude: Anthropic Claude MCP provider server")
        print()
        print("Usage:")
        print("  python -m aurarouter_claude          Start the MCP server")
        print("  python -m aurarouter_claude --help    Show this help message")
        print()
        print("Environment variables:")
        print("  ANTHROPIC_API_KEY    Your Anthropic API key (required)")
        print()
        print("MCP tools exposed:")
        print("  provider.generate              Single-shot text generation")
        print("  provider.list_models           List available Claude models")
        print("  provider.generate_with_history Multi-turn conversation")
        print("  provider.capabilities          Advertise provider features")
        sys.exit(0)

    main()
