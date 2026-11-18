# Debugging with test evidence

The Sol/Sonnet/Noir hand supplies observations. Astra/Opus owns code analysis,
interpretation, hypothesis selection and the fix. A failed assertion is evidence;
a test hand's proposed root cause is an unverified candidate.

Capture the revision and relevant uncommitted diff, exact argv and cwd, environment
versions that affect reproduction, resource allocation, random seed where relevant,
exit status, test IDs, stack/error excerpt, and raw-output path. Label **pass**,
**fail**, **timeout**, **skipped** and **blocked by environment** explicitly. An
exit-zero run with no selected tests or important skips does not establish coverage.

The brain reads the corresponding code, checks the contract and follows the data
or control path implicated by the evidence. Pick a minimal reproducer and one
discriminating check before editing. Distinguish dependency/auth/network/fixture
failure from a product bug. Record what the evidence does not establish.

For the user's GPU and inference work, add only relevant facts: tensor shape/dtype,
device/runtime version, launch parameters, numerical tolerance/reference, seed,
rank/world size and failing rank, streams/synchronization, or the timing boundary.
Serialize tests sharing a GPU or collective group; don't kill somebody else's job
to obtain a clean benchmark. Use the applicable kernel/framework skill for detailed
domain checks rather than inventing a generic diagnostic checklist.

After fixing, rerun the exact failing input and adjacent regressions on the new
revision. Add a regression when it distinguishes the bug from the old behavior;
do not add wording-only or implementation-mirroring tests. A hand reports any
missing coverage. The brain verifies results and actual diff, with no automatic
extra reruns once evidence is sufficient.

Noir timeout, malformed report or unavailable model: inspect the actual artifact,
then use Sol/Sonnet for test execution if authorized. A product assertion failure
goes directly to Astra/Opus analysis; do not replay it across model tiers. Two
failures for the same environmental cause trigger correction or a specific blocker
report. Do not auto-install dependencies, change credentials or start broad scans.
