# aurarouter-claude

Anthropic Claude provider for [AuraRouter](https://github.com/auracore-dynamics/aurarouter) -- exposes Claude models as an MCP provider server.

## Installation

```bash
pip install aurarouter-claude
```

## Usage

### As an MCP server (standalone)

```bash
python -m aurarouter_claude
```

### With AuraRouter

Once installed, AuraRouter automatically discovers this package via the `aurarouter.providers` entry point. Configure it in `auraconfig.yaml`:

```yaml
provider_catalog:
  manual:
    - name: claude
      endpoint: http://localhost:9100
```

## Configuration

Set your Anthropic API key via environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Available Models

| Model | Context Window | Input Cost | Output Cost |
|-------|---------------|------------|-------------|
| Claude Opus 4 | 200k | $15.00/1M | $75.00/1M |
| Claude Sonnet 4 | 200k | $3.00/1M | $15.00/1M |
| Claude Haiku 4.5 | 200k | $0.80/1M | $4.00/1M |

## License

Apache-2.0
