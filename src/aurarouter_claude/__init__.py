"""aurarouter-claude -- Anthropic Claude MCP provider for AuraRouter."""

__version__ = "0.5.1"
__package_name__ = "aurarouter-claude"


def get_provider_metadata():
    """Entry point callable for aurarouter.providers discovery.

    Returns a ProviderMetadata-compatible object that the
    ProviderCatalog can use to register and launch this provider.
    """
    # Import here to avoid hard dependency on aurarouter at import time.
    # We return a plain dict-like dataclass that matches ProviderMetadata.
    from dataclasses import dataclass, field

    @dataclass
    class _Metadata:
        name: str = "claude"
        provider_type: str = "mcp"
        version: str = __version__
        description: str = "Anthropic Claude models (Opus 4, Sonnet 4, Haiku 4.5)"
        command: list[str] = field(
            default_factory=lambda: ["python", "-m", "aurarouter_claude"]
        )
        requires_config: list[str] = field(
            default_factory=lambda: ["api_key"]
        )
        homepage: str = "https://github.com/auracore-dynamics/aurarouter-claude"

    return _Metadata()
