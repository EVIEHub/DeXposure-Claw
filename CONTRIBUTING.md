# Contributing

DeXposure Claw accepts changes that keep the package usable from a fresh clone.

Before opening a pull request, run these checks from the repository root:

```bash
python -m compileall src/dexposure_claw
python -m unittest discover -s tests
npm run pack:check
```

For runtime changes, include the observable command that proves the behavior.
For MCP changes, include the JSON-RPC method name, tool name, expected result
field, and failure condition in the pull request description.

The current package boundary is agent-runtime setup, MCP connectivity, and
benchmark catalog discovery. Pull requests that claim benchmark execution or
report generation should include a fresh-clone command that produces the
expected artifact path.
