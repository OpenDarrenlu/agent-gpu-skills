---
name: amd-instinct-isa
description: >-
  Local, multi-generation AMD Instinct CDNA instruction-set reference. Use for
  exact CDNA1, CDNA2, CDNA3, CDNA4, or CDNA5 ISA questions, AMD GPU assembly,
  instruction syntax and encodings, registers, wavefront execution, LDS,
  memory ordering, atomics, barriers, and matrix instructions. Downloads
  official AMD PDFs only after explicit acceptance of their document terms,
  keeps them in a Git-ignored local cache, and reports PDF page numbers.
triggers:
  - "AMD Instinct ISA"
  - "AMD GPU ISA"
  - "CDNA1"
  - "CDNA2"
  - "CDNA3"
  - "CDNA4"
  - "CDNA5"
  - "gfx950"
  - "ROCm ISA"
  - "AMD GPU assembly"
  - "AMD 指令集"
  - "CDNA 指令"
---

# AMD Instinct Multi-generation ISA Reference

Use this skill as the AMD Instinct counterpart to `cuda-skill`'s PTX
reference. It manages a local catalog of official CDNA ISA PDFs and related
architecture papers without redistributing them in this repository.

## Document terms

AMD ISA documents can include a Specification Agreement and document-specific
copyright or redistribution terms. Before downloading, review the official
document and confirm that you accept its terms. PDFs and extracted text must
not be committed or redistributed from this repository.

Every download command requires the explicit
`--accept-document-terms` flag. Files are stored only under `references/`,
which is ignored by Git.

## List the catalog

```bash
python3 scripts/amd_instinct_isa.py list
```

The checked-in catalog covers CDNA1 through CDNA5 ISA documents and the AMD
CDNA architecture white paper. Use the broad `amd-gpu-docs` skill to discover
new ROCm/Instinct documentation, then add a newly published official PDF with
`--url` without waiting for a catalog update.

## Download

Download one generation:

```bash
python3 scripts/amd_instinct_isa.py download cdna5-isa \
  --accept-document-terms
```

Download every catalog entry:

```bash
python3 scripts/amd_instinct_isa.py download --all \
  --accept-document-terms
```

Register and download a newly published official AMD PDF:

```bash
python3 scripts/amd_instinct_isa.py download \
  --url https://www.amd.com/path/to/new-isa.pdf \
  --id cdna6-isa --title "AMD Instinct CDNA6 ISA" \
  --accept-document-terms
```

Only HTTPS PDF URLs on the official `www.amd.com` or `docs.amd.com` hosts are
accepted. Use `--force` to refresh an existing local document.

## Query

Literal text query across all downloaded documents:

```bash
python3 scripts/amd_instinct_isa.py query "V_MFMA"
```

Restrict the query to CDNA5 or use a regular expression:

```bash
python3 scripts/amd_instinct_isa.py query "EXEC" --document cdna5-isa
python3 scripts/amd_instinct_isa.py query 'WMMA|matrix' --regex \
  --document cdna5-isa
```

The tool prefers `pdftotext` and falls back to Python `pdfplumber`. It reports
the official URL, local file, and PDF viewer page. Inspect the original page
for tables, diagrams, bit fields, or encodings; extracted text can lose layout.

## Answering workflow

1. Identify the exact CDNA generation before comparing instruction behavior.
2. Query the matching local PDF and read the complete surrounding section.
3. Cite the official URL, document title, and PDF page.
4. Preserve operand order, register classes, wavefront width, alignment,
   datatypes, rounding, saturation, memory ordering, and encoding fields
   exactly as written.
5. Cross-check compiler availability against current ROCm/LLVM documentation.
   ISA presence does not prove that an assembler or intrinsic exposes it.
6. Do not infer AMD semantics from an NVIDIA PTX mnemonic; make comparisons
   only after stating the AMD-native behavior.

For broad HIP APIs, ROCm installation, libraries, tools, compatibility, and
release-sensitive documentation, use `amd-gpu-docs` instead.
