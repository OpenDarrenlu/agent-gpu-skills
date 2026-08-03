---
name: amd-instinct-cdna4-isa
description: >-
  Answers questions about the AMD Instinct CDNA4 Instruction Set Architecture
  Reference Guide, including AMD GPU ISA instructions, registers, wavefront
  execution, LDS, memory operations, barriers, atomics, matrix/tensor
  operations, exceptions, and encoding details. Use this skill when the user
  mentions CDNA4, gfx950, AMD Instinct ISA, AMD GPU assembly, ROCm ISA, or asks
  to look up a detail in the official CDNA4 ISA PDF. This is a local-reference
  skill: the AMD PDF is fetched from its official URL only after the user opts
  in, and is not redistributed with this repository.
triggers:
  - "AMD Instinct ISA"
  - "AMD GPU ISA"
  - "CDNA4"
  - "CDNA4 ISA"
  - "gfx950"
  - "ROCm ISA"
  - "AMD GPU assembly"
  - "AMD 指令集"
  - "CDNA4 指令"
  - "CDNA4 架构"
---

# AMD Instinct CDNA4 ISA Reference

> Compatibility skill: new multi-generation work should use
> `amd-instinct-isa`. This directory remains installed so existing CDNA4
> commands and references keep working.

This skill is the CDNA4 counterpart to `cuda-skill`'s PTX reference. It is
designed for precise lookups in AMD's **CDNA4 Instruction Set Architecture
Reference Guide** (dated 5-August-2025), not as a replacement for ROCm/HIP
programming guidance.

## Source and licensing

The official source is:

<https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>

The document contains AMD's Specification Agreement and copyright notices.
Do not commit, mirror, quote at length, or redistribute the PDF or an extracted
copy from this repository. Run `download-cdna4-isa.sh` only if you have
accepted AMD's terms and are permitted to keep a local reference copy.

## Prepare a local reference

From this skill's installed directory (or the repository checkout), run:

```bash
bash download-cdna4-isa.sh
```

The PDF is saved under `references/` and remains local/ignored by Git. The
script uses the official AMD URL and supports `--force` to refresh it. If the
file is absent, explain that the user must opt in and download it before doing
an exact page-level lookup.

## Search workflow

Use the query helper so extracted text is temporary and is not retained in the
repository:

```bash
bash query-cdna4-isa.sh "EXEC|VCC|SCC"
bash query-cdna4-isa.sh "v_mfma"
bash query-cdna4-isa.sh "LDS"
```

The helper prefers `pdftotext` (for example, `brew install poppler` on macOS
or `apt install poppler-utils` on Debian/Ubuntu) and falls back to Python's
`pdfplumber` when available. It prints matching lines and PDF page numbers;
inspect the original page in a PDF viewer when a table or encoding diagram is
involved.

## Answering rules

1. Search the local PDF before answering an ISA-specific question. If it is not
   available, say so and provide the exact download command rather than
   inventing an opcode or operand rule.
2. Report the PDF page number/section and distinguish architectural guarantees
   from compiler/ROCm conventions. The PDF's printed page number may differ
   from the PDF viewer page because the front matter contains the agreement.
3. Preserve operand order, register classes (SGPR/VGPR/accumulator), wavefront
   width, alignment, datatype, rounding, saturation, and memory-ordering
   qualifiers exactly as specified.
4. For code-generation questions, cross-check the ISA result against the
   installed ROCm assembler/compiler documentation; ISA syntax alone does not
   guarantee that a particular ROCm release exposes an instruction.
5. Do not infer NVIDIA PTX names or semantics from an AMD mnemonic. Explain
   analogies only after stating the AMD-native behavior.

## Scope

This reference can answer, when the relevant section is present in the local
PDF:

- instruction syntax, operands, modifiers, encodings, and per-instruction
  behavior;
- scalar/vector/accumulator registers, EXEC/VCC/SCC state, wavefront and
  workgroup behavior;
- LDS, global/flat memory operations, cache/ordering qualifiers, atomics,
  barriers, exceptions, and synchronization;
- CDNA4 matrix/tensor instructions and datatype restrictions;
- comparisons with HIP/LLVM/ROCm only when the companion toolchain docs are
  available.

It does not by itself provide a complete HIP API, ROCm installation guide, or
performance model. Route those questions to the AMD upstream skills installed
from `repos/amd-skills/` or to the relevant ROCm documentation.
