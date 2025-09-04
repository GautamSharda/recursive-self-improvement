Measure codex's ability to execute on arc-agi without any additional scaffolding provided outside of the rsimp repository at this commit.

1) Build /codex from source following rsimp/experiments/arc-agi/codex-baseline/codex/docs/install.md#build-from-source
---You may have to install pkg-config and OpenSSL development libraries on Ubuntu `apt install pkg-config libssl-dev` (and this may be an issue in openai/codex)

2) Prompt it to read the README's at rsimp/ and codex-baseline/ (this) and carry out the experiment

Result: Ran out of tokens very quickly, need higher quota...