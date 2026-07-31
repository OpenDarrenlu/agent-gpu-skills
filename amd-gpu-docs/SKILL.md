---
name: amd-gpu-docs
description: >-
  Local, searchable AMD technical documentation skill for ROCm, HIP, AMD
  Instinct, ROCm libraries and tools, AMD GPU architecture, Ryzen AI, Quark,
  ROCm blogs, GPUOpen, and the installed amd/skills catalog. Use when the user
  asks for AMD or ROCm documentation, APIs, programming guides, library/tool
  behavior, architecture references, compatibility, installation, examples,
  or wants to download, index, search, compare, or study AMD technical
  material locally. For exact CDNA4 opcode and encoding questions, also use
  amd-instinct-cdna4-isa.
triggers:
  - "AMD documentation"
  - "AMD docs"
  - "AMD GPU docs"
  - "ROCm documentation"
  - "ROCm docs"
  - "HIP documentation"
  - "AMD Instinct documentation"
  - "GPUOpen"
  - "Ryzen AI docs"
  - "Quark docs"
  - "AMD 资料"
  - "ROCm 文档"
  - "AMD 本地知识库"
---

# AMD GPU Documentation Knowledge Base

Use this skill for broad AMD GPU/ROCm documentation questions. It maintains a
local, ignored cache of official public technical documentation and queries
that cache together with the installed `amd/skills` catalog.

## Scope and the meaning of “all”

AMD publishes multiple product generations, historical versions, marketing
pages, support articles, and documents governed by different agreements.
“All” in this skill means the current public technical catalogs relevant to
GPU developers:

- AMD ROCm documentation and linked component projects (HIP, math libraries,
  communication libraries, compilers, profilers, debuggers, and management
  tools);
- AMD Instinct data-center GPU documentation;
- in the `all` profile: ROCm blogs, Ryzen AI docs, Quark docs, and GPUOpen;
- the local `repos/amd-skills/skills/` catalog;
- optionally, official AMD PDFs discovered from those pages.

It intentionally excludes private/support-portal content, product marketing,
community posts, unrelated CPU/FPGA material, hidden preview versions blocked
by `robots.txt`, and arbitrary third-party pages. Historical documentation is
not fetched unless a future source profile explicitly lists it.

## Synchronize locally

Core GPU/ROCm/Instinct documentation:

```bash
python3 scripts/sync_amd_docs.py --profile core
```

All current technical catalogs:

```bash
python3 scripts/sync_amd_docs.py --profile all
```

Refresh pages already present in the cache:

```bash
python3 scripts/sync_amd_docs.py --profile all --refresh
```

PDFs are opt-in because AMD documents can contain document-specific
Specification Agreements or redistribution restrictions:

```bash
python3 scripts/sync_amd_docs.py --profile all \
  --include-pdfs --accept-document-terms
```

The downloader only creates files under `cache/`, which is ignored by Git.
Never commit or redistribute the cache without reviewing every source license.

Useful controls:

```bash
# Preview configured sources without downloading
python3 scripts/sync_amd_docs.py --profile all --list-sources

# Small validation crawl
python3 scripts/sync_amd_docs.py --profile core --max-pages 50

# Increase/decrease network parallelism
python3 scripts/sync_amd_docs.py --profile all --workers 8
```

## Query

```bash
python3 scripts/query_amd_docs.py "hipGraph"
python3 scripts/query_amd_docs.py "gfx950" --source instinct
python3 scripts/query_amd_docs.py "memory ordering" --limit 40
python3 scripts/query_amd_docs.py 'V_MFMA|matrix core' --regex
```

To include locally downloaded PDFs in a query (slower):

```bash
python3 scripts/query_amd_docs.py "MFMA" --include-pdfs
```

The query tool reports the official source URL, local file, line or PDF page,
and surrounding context. Prefer source Markdown/reStructuredText over rendered
HTML fallback text.

## Answering workflow

1. Run a narrow query using the most specific API, instruction, error, or
   architecture term.
2. Read the matching local source file around the reported lines. Do not rely
   only on the short query excerpt.
3. Cite the official source URL and version path. Treat `latest` as mutable and
   mention the local sync timestamp when version sensitivity matters.
4. Cross-check related sources for compiler/runtime versus hardware semantics.
   An ISA document, HIP API page, and ROCm library guide answer different
   layers of the same question.
5. For performance claims, use documentation to form a hypothesis, then ask
   for the smallest relevant profile or benchmark. Documentation alone is not
   workload evidence.
6. For exact CDNA4 opcode/encoding details, invoke
   `amd-instinct-cdna4-isa/query-cdna4-isa.sh` and report the PDF page.

## Local layout

```text
amd-gpu-docs/
├── SKILL.md
├── sources.json
├── scripts/
│   ├── sync_amd_docs.py
│   └── query_amd_docs.py
└── cache/                 # generated locally; Git ignored
    ├── html/              # navigation pages used for discovery
    ├── sources/           # original Sphinx Markdown/rST where available
    ├── text/              # readable fallback extracted from HTML
    ├── pdfs/              # opt-in official PDFs
    ├── manifest.json      # URL ↔ local file mapping
    └── INDEX.md           # synchronization summary
```

If the cache does not exist, explain which profile is needed and run the
smallest synchronization that covers the question.
