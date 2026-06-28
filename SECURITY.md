# Security

DeXposure Claw does not execute trades, sign transactions, or custody assets.
The current package boundary is local CLI execution, MCP stdio messages, and
runtime-adapter generation.

Report security issues through GitHub issues when the report contains no
secret values. If the report requires non-public details, first open a minimal
issue that states the affected object and observable failure condition without
including tokens, private keys, wallet seeds, credentials, or unpublished data.

Supported version:

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

When reporting a vulnerability, include:

- Affected command, MCP method, file path, or package artifact.
- Reproduction steps from a fresh clone.
- Expected result field or exit code.
- Actual result field or exit code.
- Whether any secret material was exposed.
