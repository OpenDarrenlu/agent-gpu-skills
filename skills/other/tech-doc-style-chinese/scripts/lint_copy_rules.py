#!/usr/bin/env python3
"""轻量中文技术文案检查器。

检查结果分为三类：
- error：高度确定的错误，默认导致非零退出。
- warning：依赖语境的可疑表达，需人工判断。
- style：团队风格或术语偏好，需结合项目规范判断。

脚本忽略 Markdown front matter、代码块、行内代码、URL、链接目标和
常见 API 路径。行尾加入 ``<!-- copy-lint-disable-line -->`` 可以忽略该行。

``--profile gpu`` 在上游规则之外启用 GPU / AI Infra 技术报告规则：
领域术语大小写、数字与单位留白、加速比与尺寸写法、绝对化与模糊量化表述、
领域错词与语境词，以及同一文档内的术语一致性。默认 ``--profile base``
与上游行为一致。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGETS = ["."]
INLINE_IGNORE_MARKER = "<!-- copy-lint-disable-line -->"
SKIP_DIR_NAMES = {".git", ".venv", "node_modules", "vendor"}

FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
QUOTE_PREFIX_RE = re.compile(r"^ {0,3}>[ \t]?")
LIST_ITEM_RE = re.compile(r"^( *)(?:[-+*]|\d{1,9}[.)])([ \t]+)(.*)$")
HTML_TAG_RE = re.compile(
    r'''</?[A-Za-z][A-Za-z0-9:-]*(?=[\s/>])(?:[^<>"']|"[^"]*"|'[^']*')*>'''
)
AUTOLINK_RE = re.compile(r"<https?://[^<>\s]+>")
INLINE_CODE_RE = re.compile(r"(`+)(.*?)\1")
BLOCK_BOUNDARY_RE = re.compile(
    r"(?m)^ *(?:$|#{1,6}(?:\s|$)|`{3,}|~{3,}|(?:[-*_] *){3,}$)"
)


def mask_code_spans(source: str) -> str:
    """Mask exact backtick pairs without losing diagnostic coordinates."""
    chars = list(source)
    boundaries = [match.start() for match in BLOCK_BOUNDARY_RE.finditer(source)]
    boundaries = sorted({0, len(source), *boundaries})
    for start, end in zip(boundaries, boundaries[1:]):
        runs = list(re.finditer(r"`+", source[start:end]))
        index = 0
        while index < len(runs):
            opening = runs[index]
            before = source[start:start + opening.start()]
            if (len(before) - len(before.rstrip("\\"))) % 2:
                index += 1
                continue
            closing_index = next((j for j in range(index + 1, len(runs))
                                  if runs[j].group() == opening.group()), None)
            if closing_index is None:
                index += 1
                continue
            for pos in range(start + opening.start(), start + runs[closing_index].end()):
                if chars[pos] not in "\r\n":
                    chars[pos] = " "
            index = closing_index + 1
    return "".join(chars)


URL_RE = re.compile(r'''https?://[^\s<>"'，。；：！？、（）「」『』【】]+''')
API_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/[A-Za-z0-9._~%-]+"
    r"(?:/[A-Za-z0-9._~%-]+)*(?:\?[^\s)]*)?(?![A-Za-z0-9_])"
)
INLINE_LINK_RE = re.compile(r"(!?\[[^\]]*]\()([^)]+)(\))")

FORBIDDEN_QUOTES = {
    '"': "ASCII 双引号",
    "“": "中文弯引号",
    "”": "中文弯引号",
}

NON_WORD_CHARS = r"\u4e00-\u9fffA-Za-z0-9_"
PREFIX_CONTEXT_CHARS = "与跟对向给帮替为请让"
SUFFIX_HINTS = "可会要能应需请把将来去做看读写用"

FORBIDDEN_ADDRESS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            rf"(?<![{NON_WORD_CHARS}])你(?=$|[^\u4e00-\u9fff]|[{SUFFIX_HINTS}])"
        ),
        "你",
    ),
    (re.compile(rf"(?<=[{PREFIX_CONTEXT_CHARS}])你"), "你"),
    (re.compile(r"您"), "您"),
    (re.compile(r"同学(?:们|們)?"), "同学"),
]

CASE_RULES = [
    (re.compile(r"(?<![A-Za-z0-9_])(?:id|Id)(?![A-Za-z0-9_])"), "ID"),
    (re.compile(r"(?<![A-Za-z0-9_])(?:http|Http)(?![A-Za-z0-9_])"), "HTTP"),
    (re.compile(r"(?<![A-Za-z0-9_])(?:url|Url)(?![A-Za-z0-9_])"), "URL"),
    (re.compile(r"(?<![A-Za-z0-9_])(?:json|Json)(?![A-Za-z0-9_])"), "JSON"),
    (re.compile(r"(?<![A-Za-z0-9_])(?:api|Api)(?![A-Za-z0-9_])"), "API"),
    (re.compile(r"(?<![A-Za-z0-9_])(?:ai|Ai)(?![A-Za-z0-9_])"), "AI"),
]

ABBREVIATION_RULES = [
    (re.compile(r"(?<![A-Za-z0-9_])(?:JS|Js)(?![A-Za-z0-9_])"), "JavaScript"),
    (re.compile(r"(?<![A-Za-z0-9_])H5(?![A-Za-z0-9_])"), "移动 Web 页面或 HTML5"),
]

AI_TERM_RULES = [
    (re.compile(r"(?<![A-Za-z0-9_])(?:llm|Llm)(?![A-Za-z0-9_])"), "LLM"),
    (re.compile(r"(?<![A-Za-z0-9_])(?:aigc|Aigc)(?![A-Za-z0-9_])"), "AIGC"),
    (re.compile(r"(?<![A-Za-z0-9_])(?:rag|Rag)(?![A-Za-z0-9_])"), "RAG"),
    (
        re.compile(r"(?<![A-Za-z0-9_])(?:chatgpt|Chatgpt)(?![A-Za-z0-9_])"),
        "ChatGPT",
    ),
    (
        re.compile(
            r"(?<![A-Za-z0-9_])(?:openai|OpenAI)\s+(?:api|Api)(?![A-Za-z0-9_])"
        ),
        "OpenAI API",
    ),
    (re.compile(r"(?<![A-Za-z0-9_])embeding(?![A-Za-z0-9_])"), "embedding"),
    (re.compile(r"(?<![A-Za-z0-9_])finetune(?![A-Za-z0-9_])"), "fine-tuning"),
    (
        re.compile(r"(?<![A-Za-z0-9_])fine\s+tune(?![A-Za-z0-9_])"),
        "fine-tuning",
    ),
    (re.compile(r"提示工程学"), "提示工程"),
    (re.compile(r"模型出现幻听"), "模型出现幻觉"),
]

ERROR_TYPO_RULES = [
    (re.compile(r"阀值"), "阈值"),
    (re.compile(r"布署"), "部署"),
    (re.compile(r"反回"), "返回"),
    (re.compile(r"回朔"), "回溯"),
    (re.compile(r"做为"), "作为"),
]

CONTEXT_WARNING_RULES = [
    (re.compile(r"登陆"), "确认语义：登录系统；登陆陆地或天体"),
    (re.compile(r"配制"), "确认语义：配置参数；配制溶液"),
    (re.compile(r"起用"), "确认语义：启用功能；起用人员"),
    (re.compile(r"标示"), "确认语义：标识字段；标示位置"),
    (re.compile(r"帐户"), "项目可统一为账户"),
    (re.compile(r"帐号"), "项目可统一为账号"),
    (re.compile(r"截止"), "确认语义：截至某时；截止日期"),
    (re.compile(r"搜寻"), "技术文档通常使用搜索"),
    (
        re.compile(
            r"即时(?=(通信|消息|处理|监控|更新|同步|计算|响应|数据|日志|推送|渲染|指标|告警|检索|查询|任务|分析|流式|调用|服务|接口|系统))"
        ),
        "技术能力语境通常使用实时",
    ),
]


# ---------------------------------------------------------------------------
# GPU / AI Infra 技术报告规则（--profile gpu）。
# 规则只作用于可见正文；代码、URL、路径已在 prepare_visible_line 中屏蔽。
# ---------------------------------------------------------------------------

PROFILES = ("base", "gpu")

# 术语边界：排除标识符的一部分，例如 torch.cuda、sm_90、cute::make_layout、triton.jit。
GPU_WORD_BEFORE = r"(?<![A-Za-z0-9_.])"
GPU_WORD_AFTER = r"(?![A-Za-z0-9_]|::|\.[A-Za-z_])"


def gpu_word(pattern: str) -> re.Pattern[str]:
    return re.compile(f"{GPU_WORD_BEFORE}(?:{pattern}){GPU_WORD_AFTER}")


GPU_CASE_RULES = [
    (gpu_word(r"cuda|Cuda"), "CUDA"),
    (gpu_word(r"ptx|Ptx"), "PTX"),
    (gpu_word(r"sass|Sass"), "SASS"),
    (gpu_word(r"nccl|Nccl"), "NCCL"),
    (gpu_word(r"nvshmem|Nvshmem"), "NVSHMEM"),
    (gpu_word(r"cutlass|Cutlass"), "CUTLASS"),
    (gpu_word(r"cute|Cute"), "CuTe"),
    (gpu_word(r"triton"), "Triton"),
    (gpu_word(r"pytorch|Pytorch"), "PyTorch"),
    (gpu_word(r"sglang|Sglang|SGlang"), "SGLang"),
    (gpu_word(r"vllm|VLLM|Vllm"), "vLLM"),
    (
        gpu_word(r"flash[ -]?attention|Flash[ -][Aa]ttention|Flashattention"),
        "FlashAttention",
    ),
    (gpu_word(r"flashinfer|Flashinfer"), "FlashInfer"),
    (gpu_word(r"deepgemm|DeepGemm|Deepgemm"), "DeepGEMM"),
    (gpu_word(r"deepep|DeepEp|Deepep"), "DeepEP"),
    (gpu_word(r"nvlink|Nvlink|NVlink"), "NVLink"),
    (gpu_word(r"infiniband|Infiniband"), "InfiniBand"),
    (gpu_word(r"gpu|Gpu"), "GPU"),
    (gpu_word(r"cpu|Cpu"), "CPU"),
    (gpu_word(r"hbm|Hbm"), "HBM"),
    (gpu_word(r"smem|Smem"), "SMEM"),
    (gpu_word(r"tma|Tma"), "TMA"),
    (gpu_word(r"gemm|Gemm"), "GEMM"),
    (gpu_word(r"moe|Moe|MOE"), "MoE"),
    (gpu_word(r"fp8|Fp8"), "FP8"),
    (gpu_word(r"bf16|Bf16"), "BF16"),
    (gpu_word(r"fp16|Fp16"), "FP16"),
    (gpu_word(r"fp32|Fp32"), "FP32"),
    (gpu_word(r"fp4|Fp4"), "FP4"),
    (gpu_word(r"nvfp4|Nvfp4"), "NVFP4"),
    (gpu_word(r"int8|Int8"), "INT8"),
    (gpu_word(r"tensor ?cores?|Tensor ?cores?|TensorCores?"), "Tensor Core"),
    (
        gpu_word(r"kv[ -]?cache|Kv[ -]?[Cc]ache|KV[ -]Cache|KVCache|KV-cache|kvcache"),
        "KV cache",
    ),
    (gpu_word(r"hopper|blackwell|ampere"), "架构名首字母大写：Hopper、Blackwell、Ampere"),
    (gpu_word(r"nsight compute|Nsight compute"), "Nsight Compute"),
    (gpu_word(r"nsight systems|Nsight systems"), "Nsight Systems"),
    (gpu_word(r"sm"), "SM"),
    (gpu_word(r"TFlops|Tflops|tflops"), "TFLOPS"),
    (gpu_word(r"GFlops|Gflops|gflops|PFlops|Pflops|pflops"), "GFLOPS 或 PFLOPS"),
]

GPU_TERM_PREFERENCE_RULES = [
    (re.compile(r"线程束"), "约定使用 warp"),
    (re.compile(r"张量核心?"), "约定使用 Tensor Core"),
    (re.compile(r"核函数"), "约定使用 kernel"),
    (re.compile(r"内核"), "GPU 语境写 kernel；「内核」易与操作系统内核混淆"),
    (re.compile(r"线程块"), "约定使用 CTA，首次出现说明即 thread block"),
    (re.compile(r"占用率"), "occupancy 保留英文，避免与利用率混淆"),
]

GPU_ERROR_TYPO_RULES = [
    (re.compile(r"瓶劲"), "瓶颈"),
    (re.compile(r"阻赛"), "阻塞"),
    (re.compile(r"兼融"), "兼容"),
    (re.compile(r"冗于"), "冗余"),
    (re.compile(r"操做"), "操作"),
]

GPU_CONTEXT_WARNING_RULES = [
    (re.compile(r"规约"), "确认语义：reduction 写「归约」；「规约」指规范、协议"),
    (re.compile(r"通讯"), "技术文档通常写「通信」"),
    (re.compile(r"(?:内存|显存)泄露"), "通常写「泄漏」；「泄露」用于信息泄露"),
    (re.compile(r"重迭"), "通常写「重叠」"),
]

GPU_OVERCLAIM_RE = re.compile(
    r"零开销|完全掩盖|完全隐藏|完美(?:重叠|掩盖|隐藏|扩展)|线性扩展|性能翻倍|"
    r"吊打|遥遥领先|业界领先|业界最快|极致性能|(?<![A-Za-z])SOTA(?![A-Za-z])"
)
GPU_VAGUE_RE = re.compile(
    r"(?:显著|大幅|明显|极大|大大)地?"
    r"(?:提升|提高|降低|减少|加速|优化|改善|缩短|下降)"
)

GPU_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)"
    r"(ms|us|µs|μs|ns|[KMGTP]i?B(?:/s)?|[GTP]FLOPS|[GM]Hz)"
    r"(?![A-Za-z0-9_])"
)
# 中文与半角英文、数字之间缺少空格（盘古之白）。使用前瞻以捕获相邻的多处边界。
GPU_CJK_SPACING_RE = re.compile(
    r"(?=([\u4e00-\u9fff][A-Za-z0-9]|[A-Za-z0-9][\u4e00-\u9fff]))"
)
GPU_SPEEDUP_RE = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?[xX](?![A-Za-z0-9_])")
GPU_SHAPE_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?!0[xX])\d+(?:[xX]\d+)+(?![A-Za-z0-9_])"
)

# 同一文档内应统一的写法：(组名, [(显示名, 正则), ...])
GPU_CONSISTENCY_GROUPS: list[tuple[str, list[tuple[str, re.Pattern[str]]]]] = [
    ("bubble", [("空泡", re.compile(r"空泡")), ("气泡", re.compile(r"气泡"))]),
    ("latency", [("延迟", re.compile(r"延迟")), ("时延", re.compile(r"时延"))]),
    (
        "arithmetic intensity",
        [
            ("算术强度", re.compile(r"算术强度")),
            ("计算强度", re.compile(r"计算强度")),
            ("计算访存比", re.compile(r"计算访存比")),
        ],
    ),
    ("dequantization", [("反量化", re.compile(r"反量化")), ("解量化", re.compile(r"解量化"))]),
    ("pipeline", [("流水线", re.compile(r"流水线")), ("管线", re.compile(r"管线"))]),
    (
        "microsecond",
        [
            ("µs", re.compile(r"\d ?[µμ]s(?![A-Za-z])")),
            ("us", re.compile(r"\d ?us(?![A-Za-z])")),
        ],
    ),
]


@dataclass(frozen=True)
class Violation:
    file: Path
    line: int
    col: int
    severity: str
    kind: str
    message: str
    snippet: str


def mask_match(text: str, regex: re.Pattern[str]) -> str:
    return regex.sub(lambda match: " " * (match.end() - match.start()), text)


def prepare_visible_line(line: str) -> str:
    visible = mask_match(line, INLINE_CODE_RE)
    visible = mask_match(visible, AUTOLINK_RE)
    visible = INLINE_LINK_RE.sub(
        lambda match: (
            f"{match.group(1)}{' ' * len(match.group(2))}{match.group(3)}"
        ),
        visible,
    )
    visible = mask_match(visible, URL_RE)
    visible = mask_match(visible, API_PATH_RE)
    return visible


def iter_forbidden_address_matches(line: str):
    seen: set[tuple[int, int, str]] = set()
    for pattern, label in FORBIDDEN_ADDRESS_PATTERNS:
        for match in pattern.finditer(line):
            key = (match.start(), match.end(), label)
            if key not in seen:
                seen.add(key)
                yield match, label


def add_rule_matches(
    violations: list[Violation],
    *,
    path: Path,
    line_no: int,
    raw: str,
    visible: str,
    rules: list[tuple[re.Pattern[str], str]],
    severity: str,
    kind: str,
    label: str,
) -> None:
    for pattern, suggested in rules:
        for match in pattern.finditer(visible):
            wrong = match.group(0)
            violations.append(
                Violation(
                    file=path,
                    line=line_no,
                    col=match.start() + 1,
                    severity=severity,
                    kind=kind,
                    message=f"{label}「{wrong}」：{suggested}",
                    snippet=raw.strip(),
                )
            )


def scan_markdown(path: Path, profile: str = "base") -> list[Violation]:
    if profile not in PROFILES:
        raise ValueError(f"未知 profile: {profile}")
    violations: list[Violation] = []
    gpu_visible_lines: list[tuple[int, str, str]] = []
    fence_delimiter: str | None = None
    source = path.read_text(encoding="utf-8")
    raw_lines = source.splitlines()
    # Preserve line/column positions, including multiline HTML attributes.
    source = HTML_TAG_RE.sub(
        lambda match: re.sub(r"[^\r\n]", " ", match.group()), source
    )
    lines = mask_code_spans(source).splitlines()
    in_front_matter = bool(lines and lines[0].strip() == "---")
    previous_depth = 0
    paragraph_open = False
    indented_code = False
    list_indents: list[int] = []
    fence_depth = 0
    fence_indent = 0

    for line_no, content in enumerate(lines, start=1):
        raw = raw_lines[line_no - 1]
        # HTML masking must not turn a tag into apparent code indentation.
        content = raw
        if in_front_matter:
            if line_no != 1 and raw.strip() in {"---", "..."}:
                in_front_matter = False
            continue

        # Only consume the containers recorded at the opening fence. Symbols
        # inside code (including > and list markers) are literal content.
        if fence_delimiter is not None:
            candidate = raw.expandtabs(4)
            container_present = True
            for _ in range(fence_depth):
                prefix = QUOTE_PREFIX_RE.match(candidate)
                if prefix is None:
                    container_present = False
                    break
                candidate = candidate[prefix.end():]
            if container_present and fence_indent:
                if candidate.strip() and not candidate.startswith(" " * fence_indent):
                    container_present = False
                else:
                    candidate = candidate[fence_indent:]
            if container_present:
                closing = FENCE_RE.match(candidate)
                if (
                    closing
                    and closing.group(1)[0] == fence_delimiter[0]
                    and len(closing.group(1)) >= len(fence_delimiter)
                    and not closing.group(2).strip()
                ):
                    fence_delimiter = None
                continue
            fence_delimiter = None

        content = raw.expandtabs(4)
        depth = 0
        while prefix := QUOTE_PREFIX_RE.match(content):
            content = content[prefix.end():]
            depth += 1
        if depth != previous_depth:
            paragraph_open = False
            indented_code = False
            list_indents.clear()
        previous_depth = depth

        if not content.strip():
            paragraph_open = False
            continue
        indentation = len(content) - len(content.lstrip(" "))
        while list_indents and indentation < list_indents[-1]:
            list_indents.pop()
            paragraph_open = False
            indented_code = False
        container_indent = list_indents[-1] if list_indents else 0
        content = content[container_indent:]
        item = LIST_ITEM_RE.match(content)
        if item and len(item.group(1)) <= 3:
            # Up to four padding spaces belong to a list marker. Greater
            # padding leaves indentation that can start an indented code block.
            padding = len(item.group(2))
            marker_width = item.start(3) - (padding - 1 if padding > 4 else 0)
            container_indent += marker_width
            list_indents.append(container_indent)
            content = content[marker_width:]
            paragraph_open = False
            indented_code = False

        fence_match = FENCE_RE.match(content)
        if content.startswith(("    ", "\t")) and (
            indented_code or not paragraph_open
        ):
            indented_code = True
            continue
        indented_code = False

        if fence_match:
            delimiter = fence_match.group(1)
            if delimiter[0] != "`" or "`" not in fence_match.group(2):
                fence_delimiter = delimiter
                fence_depth = depth
                fence_indent = container_indent
                paragraph_open = False
                continue

        paragraph_open = not bool(re.match(
            r"^ {0,3}(?:#{1,6}(?:\s|$)|(?:[-*_] *){3,}$|=+\s*$)", content
        ))
        if INLINE_IGNORE_MARKER in raw:
            continue

        visible = prepare_visible_line(lines[line_no - 1])

        for quote, label in FORBIDDEN_QUOTES.items():
            for match in re.finditer(re.escape(quote), visible):
                violations.append(
                    Violation(
                        file=path,
                        line=line_no,
                        col=match.start() + 1,
                        severity="style",
                        kind="quote",
                        message=f"可见正文包含{label}，确认是否应改为直角引号「」",
                        snippet=raw.strip(),
                    )
                )

        for match, term in iter_forbidden_address_matches(visible):
            violations.append(
                Violation(
                    file=path,
                    line=line_no,
                    col=match.start() + 1,
                    severity="style",
                    kind="address",
                    message=f"可见正文包含称呼「{term}」，确认项目是否允许",
                    snippet=raw.strip(),
                )
            )

        add_rule_matches(
            violations,
            path=path,
            line_no=line_no,
            raw=raw,
            visible=visible,
            rules=CASE_RULES,
            severity="style",
            kind="casing",
            label="术语写法",
        )
        add_rule_matches(
            violations,
            path=path,
            line_no=line_no,
            raw=raw,
            visible=visible,
            rules=ABBREVIATION_RULES,
            severity="style",
            kind="abbreviation",
            label="缩写需结合语境确认",
        )
        add_rule_matches(
            violations,
            path=path,
            line_no=line_no,
            raw=raw,
            visible=visible,
            rules=AI_TERM_RULES,
            severity="warning",
            kind="ai-term",
            label="AI 术语",
        )
        add_rule_matches(
            violations,
            path=path,
            line_no=line_no,
            raw=raw,
            visible=visible,
            rules=ERROR_TYPO_RULES,
            severity="error",
            kind="typo",
            label="高置信度错词",
        )
        add_rule_matches(
            violations,
            path=path,
            line_no=line_no,
            raw=raw,
            visible=visible,
            rules=CONTEXT_WARNING_RULES,
            severity="warning",
            kind="context",
            label="语境词",
        )

        if profile == "gpu":
            gpu_visible_lines.append((line_no, raw, visible))
            scan_gpu_line(violations, path=path, line_no=line_no, raw=raw, visible=visible)

    if profile == "gpu":
        violations.extend(check_gpu_consistency(path, gpu_visible_lines))

    return violations


def add_message_matches(
    violations: list[Violation],
    *,
    path: Path,
    line_no: int,
    raw: str,
    visible: str,
    pattern: re.Pattern[str],
    severity: str,
    kind: str,
    message,
) -> None:
    for match in pattern.finditer(visible):
        violations.append(
            Violation(
                file=path,
                line=line_no,
                col=match.start() + 1,
                severity=severity,
                kind=kind,
                message=message(match),
                snippet=raw.strip(),
            )
        )


def scan_gpu_line(
    violations: list[Violation], *, path: Path, line_no: int, raw: str, visible: str
) -> None:
    common = dict(path=path, line_no=line_no, raw=raw, visible=visible)
    add_rule_matches(
        violations, **common, rules=GPU_CASE_RULES,
        severity="style", kind="gpu-casing", label="GPU 术语写法",
    )
    add_rule_matches(
        violations, **common, rules=GPU_TERM_PREFERENCE_RULES,
        severity="style", kind="gpu-term", label="术语偏好",
    )
    add_rule_matches(
        violations, **common, rules=GPU_ERROR_TYPO_RULES,
        severity="error", kind="typo", label="高置信度错词",
    )
    add_rule_matches(
        violations, **common, rules=GPU_CONTEXT_WARNING_RULES,
        severity="warning", kind="context", label="语境词",
    )
    add_message_matches(
        violations, **common, pattern=GPU_OVERCLAIM_RE,
        severity="warning", kind="overclaim",
        message=lambda m: f"绝对化表述「{m.group(0)}」：给出证据和适用条件，或改为有限表述",
    )
    add_message_matches(
        violations, **common, pattern=GPU_VAGUE_RE,
        severity="warning", kind="vague",
        message=lambda m: f"模糊量化「{m.group(0)}」：补充数值、基线和测量条件",
    )
    spacing = list(GPU_CJK_SPACING_RE.finditer(visible))
    if spacing:
        # 同一行只报告一次，避免缺少留白的旧文档淹没其他问题。
        first = spacing[0]
        example = visible[max(0, first.start() - 3):first.start() + 5].strip()
        violations.append(
            Violation(
                file=path,
                line=line_no,
                col=first.start() + 1,
                severity="style",
                kind="spacing",
                message=(
                    f"本行有 {len(spacing)} 处中文与英文、数字之间缺少半角空格，"
                    f"例如「{example}」"
                ),
                snippet=raw.strip(),
            )
        )
    add_message_matches(
        violations, **common, pattern=GPU_UNIT_RE,
        severity="style", kind="unit-spacing",
        message=lambda m: f"数字与单位之间加一个半角空格：「{m.group(1)} {m.group(2)}」",
    )
    add_message_matches(
        violations, **common, pattern=GPU_SHAPE_RE,
        severity="style", kind="shape",
        message=lambda m: (
            f"尺寸「{m.group(0)}」写作「{re.sub('[xX]', '×', m.group(0))}」，或放入行内代码"
        ),
    )
    add_message_matches(
        violations, **common, pattern=GPU_SPEEDUP_RE,
        severity="style", kind="speedup",
        message=lambda m: (
            f"倍数「{m.group(0)}」写作「{m.group(0)[:-1]}×」或「{m.group(0)[:-1]} 倍」，并写明基线"
        ),
    )


def check_gpu_consistency(
    path: Path, visible_lines: list[tuple[int, str, str]]
) -> list[Violation]:
    findings: list[Violation] = []
    for _, variants in GPU_CONSISTENCY_GROUPS:
        hits: dict[str, list[tuple[int, int, str]]] = {}
        for label, pattern in variants:
            for line_no, raw, visible in visible_lines:
                for match in pattern.finditer(visible):
                    hits.setdefault(label, []).append((line_no, match.start() + 1, raw))
        if len(hits) < 2:
            continue
        ordered = sorted(hits.items(), key=lambda item: (-len(item[1]), item[1][0][:2]))
        summary = "、".join(f"「{label}」{len(places)} 处" for label, places in ordered)
        for label, places in ordered[1:]:
            line_no, col, raw = places[0]
            findings.append(
                Violation(
                    file=path,
                    line=line_no,
                    col=col,
                    severity="style",
                    kind="consistency",
                    message=f"同一文档混用 {summary}；统一为一种写法（首次出现「{label}」）",
                    snippet=raw.strip(),
                )
            )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验中文技术文案高频规则")
    parser.add_argument(
        "files",
        nargs="*",
        help="要检查的 Markdown 文件或目录；为空时检查当前目录",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="将 warning 和 style 也作为失败处理",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="base",
        help="规则集：base 为上游通用规则；gpu 额外启用 GPU / AI Infra 技术报告规则",
    )
    return parser.parse_args()


def collect_targets(args: argparse.Namespace) -> list[Path]:
    raw_targets = args.files if args.files else DEFAULT_TARGETS
    targets: list[Path] = []

    for item in raw_targets:
        path = Path(item)
        if not path.exists():
            print(f"[WARN] 文件不存在，已跳过: {item}", file=sys.stderr)
            continue
        if path.is_dir():
            for markdown in sorted(path.rglob("*.md")):
                if any(part in SKIP_DIR_NAMES for part in markdown.parts):
                    continue
                targets.append(markdown)
        else:
            targets.append(path)

    return sorted(
        {path.resolve(): path for path in targets}.values(), key=lambda path: str(path)
    )


def main() -> int:
    args = parse_args()
    targets = collect_targets(args)
    if not targets:
        print("未找到可检查的 Markdown 文件。", file=sys.stderr)
        return 1

    findings = [
        item for target in targets for item in scan_markdown(target, args.profile)
    ]
    if not findings:
        print(f"PASS: 共检查 {len(targets)} 个文件，未发现问题。")
        return 0

    counts = {
        severity: sum(item.severity == severity for item in findings)
        for severity in ("error", "warning", "style")
    }
    for item in findings:
        print(
            f"- {item.file}:{item.line}:{item.col} "
            f"[{item.severity}/{item.kind}] {item.message}\n  {item.snippet}"
        )

    should_fail = bool(counts["error"] or (args.strict and findings))
    status = "FAIL" if should_fail else "PASS WITH ADVICE"
    print(
        f"{status}: 共检查 {len(targets)} 个文件；"
        f"error={counts['error']} warning={counts['warning']} style={counts['style']}。"
    )
    return 2 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
