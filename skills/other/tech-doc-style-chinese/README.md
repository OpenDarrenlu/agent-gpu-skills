# tech-doc-style-chinese（GPU / AI Infra 定制版）

本目录是 [Fenng/Tech-Doc-Style-Chinese](https://github.com/Fenng/Tech-Doc-Style-Chinese) 的同名本地覆盖版。上游提供中文技术文档与产品文案的通用写作规范；本版本保留上游全部通用规则，并面向算子、推理框架和训练框架方向的技术原理报告增加结构模板、术语表、证据与数字规范、图表规范、HTML 模板和 GPU 检查规则。

按仓库约定，本地 skill 优先于外部同名 skill。安装器只安装本目录，上游 clone 只用于比对和同步。

## 上游来源

- 仓库：`https://github.com/Fenng/Tech-Doc-Style-Chinese.git`
- 基线：commit `b119d01`（2026-10-05，`docs: explain STE-inspired writing methods in README`）
- 许可：MIT，版权归 Fenng，见本目录 [LICENSE](LICENSE)。本目录保留上游许可声明和作者署名。
- 本地 clone：`repos/tech-doc-style-chinese/`（git-ignored），用 `bash update-repos.sh tech-doc-style-chinese` 获取或更新。

## 文件来源

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `SKILL.md` | 修改 | 增加技术报告的适用范围、证据等级、整理模式、报告专用规则、报告类型路由、产出格式和 skill 分工 |
| `references/terminology-and-typography.md` | 原样 | 上游通用术语与排版规则 |
| `references/api-status-copy.md` | 原样 | 上游 API 状态与错误文案 |
| `references/controlled-technical-chinese.md` | 原样 | 上游受控中文技术写作 |
| `references/project-overrides-example.md` | 原样 | 上游项目覆盖模板 |
| `references/gpu-report-structures.md` | 新增 | 7 类技术原理报告的骨架、必答问题和常见问题 |
| `references/evidence-and-numbers.md` | 新增 | 证据等级、确定程度用词、性能数字上下文、加速比与峰值口径、12 个 GPU 回归样例 |
| `references/gpu-terminology.md` | 新增 | GPU、推理、训练领域的术语约定、易混淆概念、官方写法和领域错词 |
| `references/figures-formulas-code.md` | 新增 | 选图、图注、Mermaid 模式、估算公式、shape 记法和源码引用 |
| `references/output-formats.md` | 新增 | Markdown 与单文件 HTML 的选择规则、HTML 约定、截图整理流程 |
| `assets/report-template.html` | 新增 | 离线可读的单文件 HTML 报告模板 |
| `scripts/lint_copy_rules.py` | 修改 | 新增 `--profile gpu`；默认 `base` 与上游行为一致 |
| `scripts/unwrap_md_paragraphs.py` | 原样 | 展开段落硬换行 |
| `tests/test_lint_copy_rules.py`、`tests/test_unwrap_md_paragraphs.py` | 原样 | 上游回归测试 |
| `tests/test_skill_structure.py` | 修改 | 增加定制部分的结构检查 |
| `tests/test_gpu_profile_rules.py` | 新增 | GPU 规则测试 |
| `agents/openai.yaml` | 修改 | 展示名称和默认提示 |

## 检查工具

```bash
# GPU 技术报告检查：上游规则 + GPU 术语大小写、中西文留白、单位留白、加速比与尺寸写法、绝对化与模糊量化表述、领域错词、同文档术语一致性
python3 scripts/lint_copy_rules.py --profile gpu report.md

# 上游通用检查
python3 scripts/lint_copy_rules.py report.md

# 检查段落硬换行；去掉 --check 时直接写回
python3 scripts/unwrap_md_paragraphs.py --check report.md
```

不需要展开硬换行的文件，可以在独立一行写 `<!-- unwrap-disable-file -->`。

## 运行测试

```bash
cd skills/other/tech-doc-style-chinese
for t in tests/test_*.py; do python3 "$t"; done
```

## 同步上游

1. 运行 `bash update-repos.sh tech-doc-style-chinese` 更新 `repos/tech-doc-style-chinese/`。
2. 比对上游与本目录：`diff -ru repos/tech-doc-style-chinese/references skills/other/tech-doc-style-chinese/references`，以及 `SKILL.md`、`scripts/`、`tests/`。
3. 标为「原样」的文件直接覆盖；标为「修改」的文件手工合并上游改动。
4. 运行全部测试，并更新本文件中的基线 commit。

## 可调整的默认约定

- 引号：沿用上游的直角引号 `「」`。如需改为弯引号，修改 `SKILL.md` 与 `references/terminology-and-typography.md` 中的引号规则，并在 `scripts/lint_copy_rules.py` 的 `FORBIDDEN_QUOTES` 中调整。
- 首选术语：见 `references/gpu-terminology.md`。目标项目已有术语表时，以目标项目为准。
