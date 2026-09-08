# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## 仓库性质

这不是一个应用代码库，而是一个 **Agent Skill 集合（bundle）**：21 个本地 skill 目录（每个含 `SKILL.md`）+ 4 个外部 skill 仓库（git submodule，`repos/` 下）+ 若干按需稀疏检出的上游源码仓库（存放在各 skill 目录的 `repos/` 下，如 `sglang_skill/repos/sglang/`）。没有构建系统、测试套件或包管理——"开发"就是维护 bash 安装/更新脚本和各 skill 的 `SKILL.md` 及其引用资料。

## 常用命令

```bash
# 一条命令：submodule 初始化 + 外部 repo 获取 + 安装全部 skill
bash bootstrap.sh --agent claude          # 也支持 cursor/codex/gemini/kimi

# 分步：获取/更新源码 repo（triton/cutlass/deepgemm/sglang/nccl 为 sparse checkout）
bash update-repos.sh                      # 全部
bash update-repos.sh sglang               # 只更新某一个

# 安装 skill 到 agent（复制 SKILL.md，其余文件软链接回本仓库，便于 git pull 同步）
bash install.sh --agent claude
bash install.sh --agent claude --no-nvidia-skills --no-amd-skills --no-cursor-skills --no-ccfa-skills --no-veloq  # 只装本地 21 个
bash install.sh --dest /tmp/skills-test   # 隔离安装，用于验证改动不碰真实配置

# CUDA 文档抓取（cuda_skill 的 PTX/Runtime/Driver 文档，一次性或 --force 更新）
python3 scrape_docs.py all --force        # 依赖: pip install -r requirements-docs.txt

# AMD 文档知识库（amd-gpu-docs skill）
python3 amd-gpu-docs/scripts/sync_amd_docs.py --profile core
python3 amd-gpu-docs/scripts/query_amd_docs.py "hipGraph"
```

## 验证方式

没有测试套件。`install.sh` 末尾的验证段会对每个 skill 的关键文件（SKILL.md、源码 repo 关键路径、文档索引等）做存在性检查并汇总 `N 通过, M 失败`——**改脚本或更新上游 repo 后重跑 `install.sh` 看这段输出**就是主要验证手段。新增/修改 skill 时应在该验证段（`check` 调用）补充对应检查项。

## 架构要点

- **安装模型**：`install.sh` 对本地 skill **复制 SKILL.md、软链接其余文件**（Cursor 不识别软链接的 SKILL.md）。目录名与安装名不同时由 `SKILL_NAMES`/`SKILL_DIRS` 两个**平行数组**映射（如 `cuda_skill/` → `cuda-skill`）。脚本需兼容 **Bash 3.2**（macOS 自带版本），不要用关联数组等 Bash 4 特性。
- **外部 skill 来源**：`repos/` 下 4 个 submodule（NVIDIA/skills 242 个、amd/skills、Saddss/cursor-skills、mikubaka88/CCFA-Skills），由 `install.sh` 展开安装；与本地 skill 同名时本地优先（`is_local_skill`）。
- **路由分层**：`gpu-development-catchall`（兜底）→ 3 个 router（`gpu-performance-router` / `gpu-kernel-authoring-router` / `llm-serving-router`）→ 具体领域 skill。新增领域 skill 时通常要同步更新相关 router 的 `SKILL.md` 和根目录 `AGENTS.md` 的路由表。
- **sparse checkout 跟踪上游 main**：`update-repos.sh` 顶部的 `*_dirs` 数组定义各源码 repo 的稀疏检出路径。**上游重构会导致路径失效**（已发生过：sglang 的 `sgl-kernel/` 移至 `python/sglang/kernels/aot/`）——症状是 install.sh 验证段报"缺失"，修复需同步三处：`update-repos.sh` 的数组、对应 skill 的 `SKILL.md` 路径引用、`install.sh` 验证段。
- **VeloQ**：`nsys-profile-analysis` / `ncu-profile-analysis` 两个 skill 及 `veloq` 二进制由 `install-veloq.sh` 委托其官方安装器安装，本仓库不 vendored 其内容。
- **许可证边界**：AMD ISA PDF、AMD 文档页等只存本地 git-ignored 缓存，需用户显式接受条款后下载（`--accept-document-terms`），不得提交或再分发。
