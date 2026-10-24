"""README 路由契约测试。

README.md 是 agent 的执行入口（“给 Agent 的执行协议”）。这里检查它是否仍然与仓库一致：

1. 协议段落存在，并且 GPU / 非 GPU 的边界与 AGENTS.md 一致；
2. README 中提到的 skill 名称与 `$name` 都能在 inventory.tsv 中找到；
3. README 中提到的 `--选项` 确实存在于某个脚本中；
4. README 中“非 GPU”的 @README 示例，经真实路由脚本后确实选中示例所声明的 skill。

运行（仓库根目录）:
  python3 -m unittest discover -s tests -p 'test_readme_contract.py' -v
"""

import importlib.util
import os
import re
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
README = os.path.join(REPO, "README.md")
AGENTS = os.path.join(REPO, "AGENTS.md")
INVENTORY = os.path.join(REPO, "skills", "routers", "agent-skill-router", "references", "inventory.tsv")
ROUTER_SCRIPTS = os.path.join(REPO, "skills", "routers", "agent-skill-router", "scripts")

sys.path.insert(0, ROUTER_SCRIPTS)
import route_skill  # noqa: E402

OPTION_SOURCES = [
    "bootstrap.sh", "install.sh", "update-repos.sh", "install-veloq.sh", "install-external.sh",
    "install-rea-mcp.sh", os.path.join("scripts", "node-preflight.sh"),
    os.path.join("skills", "routers", "agent-skill-router", "scripts", "route_skill.py"),
    os.path.join("skills", "routers", "agent-skill-router", "scripts", "build_inventory.py"),
]

# 反引号中形如 skill 名、但实际不是 skill 的词：仓库名、目录名、launcher 标记与路由输出标签。
# 输出标签由 route_skill.py 产生（见 SKILL.md 的 Availability labels），这里只是文档词汇的白名单。
NOT_SKILL_TOKENS = {
    "agent-gpu-skills", "kernel-design-agents", "runtime-skills",
    "agent-gpu-skills-managed", "not-checked", "not-found", "user-level",
}


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def inventory_names():
    names = set()
    columns = None
    with open(INVENTORY, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if columns is None:
                columns = fields
                continue
            row = dict(zip(columns, fields))
            if row["kind"] == "skill":
                names.add(row["name"])
    return names


# README 中非 GPU 的 @README 示例 → 期望的主 skill。键是示例文本中一段稳定的子串，必须在 README 中出现。
NON_GPU_EXAMPLES = {
    "帮我检索关于GPU调度的相关论文": "ccf-literature-searcher",
    "我的 Zotero 文库里找 attention 相关的文献": "zotero-cli",
    "帮我 review 这篇投稿的论文草稿": "ccf-paper-reviewer",
    "润色这篇中文技术原理报告": "tech-doc-style-chinese",
    "用 TDD 给这个函数补测试": "tdd",
    "review 这个 PR 的代码改动": "ponytail-review",
    "审计一下这个仓库的风险代码和慢路径": "ponytail-audit",
    "把这段文字去掉 AI 味": "sepia",
    "帮我逆向分析这个 Electron 应用的 asar 包": "reverse-engineer-anything",
    "帮我总结一下我微信群昨天的聊天记录": "wx-cli",
    "用 deepstream 搭一个视频分析 pipeline": "deepstream-generate-pipeline",
    "在 BlueField DPU 上配置 DOCA DMA": "doca-dma",
    "用 Evo 2 NIM 生成一段 DNA 序列": "bionemo-evo2-nim",
    "用 earth2studio 跑一个天气预报 forecast": "earth2studio-deterministic-forecast",
    "用 cuOpt 求解一个 MILP 混合整数规划": "cuopt-numerical-optimization-api",
    "用 $tdd skill 给这个函数补测试": "tdd",
    "用 $deepstream-dev 帮我写一个 pyservicemaker 程序": "deepstream-dev",
}


class ReadmeProtocolTest(unittest.TestCase):
    def setUp(self):
        self.text = read(README)

    def test_protocol_section_exists_near_the_top(self):
        self.assertIn("## 给 Agent 的执行协议（@README 入口）", self.text)
        self.assertLess(self.text.index("## 给 Agent 的执行协议（@README 入口）"), self.text.index("## 安装"))
        for step in ("**步骤 1", "**步骤 3", "**步骤 4", "**步骤 6"):
            self.assertIn(step, self.text)

    def test_stale_catchall_for_unknown_tasks_is_gone(self):
        self.assertNotIn("不知道该用哪个 skill 时，从 `gpu-development-catchall` 开始", self.text)

    def test_gpu_boundary_matches_agents_md(self):
        agents = read(AGENTS)
        self.assertIn("For GPU-only work, start with `gpu-development-catchall`", agents)
        self.assertIn("For mixed or non-GPU work, use `agent-skill-router`", agents)
        self.assertIn("只用于 GPU 请求", self.text)
        self.assertIn("纯 GPU", self.text)
        self.assertIn("非 GPU、混合领域或不确定", self.text)

    def test_protocol_names_the_real_route_command_and_artifact_option(self):
        script = os.path.join("skills", "routers", "agent-skill-router", "scripts", "route_skill.py")
        self.assertIn("python3 " + script.replace(os.sep, "/") + " '<用户的原始任务>'", self.text)
        self.assertTrue(os.path.isfile(os.path.join(REPO, script)))
        self.assertIn("--artifact", self.text)

    def test_host_limitation_and_plain_text_fallback_are_documented(self):
        self.assertIn("不能保证每个宿主都以同样的语法加载它", self.text)
        self.assertIn("请先阅读仓库根目录的 README.md", self.text)
        self.assertIn("用户任务：<在这里写用户的原始请求>", self.text)

    def test_no_claim_that_host_auto_selection_was_measured_on_every_host(self):
        self.assertIn("没有在每个宿主上测量", self.text)


class ReadmeReferenceTest(unittest.TestCase):
    def setUp(self):
        self.text = read(README)
        self.names = inventory_names()

    def test_backticked_hyphenated_names_resolve_to_inventory(self):
        missing = []
        for token in sorted(set(re.findall(r"`([a-z][a-z0-9]*(?:-[a-z0-9]+)+)`", self.text))):
            if token in NOT_SKILL_TOKENS:
                continue
            if token not in self.names:
                missing.append(token)
        self.assertEqual(missing, [], "README names not in inventory.tsv")

    def test_dollar_skill_names_in_examples_resolve_to_inventory(self):
        names = set(re.findall(r"\$([a-z][a-z0-9]*(?:-[a-z0-9]+)+)", self.text))
        self.assertTrue(names)
        self.assertEqual(sorted(n for n in names if n not in self.names), [])

    def test_every_documented_long_option_exists_in_a_script(self):
        scripts = []
        for rel in OPTION_SOURCES:
            with open(os.path.join(REPO, rel), encoding="utf-8") as f:
                scripts.append(f.read())
        combined = "\n".join(scripts)
        options = sorted(set(re.findall(r"(?<![A-Za-z0-9-])(--[a-z][a-z0-9-]*)", self.text)))
        self.assertTrue(options)
        missing = [o for o in options if o not in combined]
        self.assertEqual(missing, [], "README options not found in any script")

    def test_documented_repo_paths_exist(self):
        for rel in (
            "skills/routers/agent-skill-router/references/inventory.tsv",
            "skills/routers/agent-skill-router/SKILL.md",
            "sources/external-sources.conf",
            "INSTALL.md",
            "docs/skills-usage-review.md",
            "tests/test_readme_contract.py",
        ):
            self.assertTrue(rel in self.text, "README does not mention " + rel)
            self.assertTrue(os.path.exists(os.path.join(REPO, rel)), "missing " + rel)


class ReadmeExampleRoutingTest(unittest.TestCase):
    """README 中的非 GPU @README 示例必须经真实路由脚本选中所声明的 skill。"""

    rules = route_skill.load_rules()
    inventory = route_skill.load_inventory()

    def test_each_documented_non_gpu_example_is_present_in_readme(self):
        text = read(README)
        for fragment in NON_GPU_EXAMPLES:
            self.assertIn(fragment, text, fragment)

    def test_each_documented_non_gpu_example_routes_to_its_skill(self):
        text = read(README)
        for fragment, expected in NON_GPU_EXAMPLES.items():
            # 只取真正的 @README 示例行，而不是正文中提到同一短语的说明。
            line = next(l for l in text.splitlines() if l.startswith("@README") and fragment in l)
            query = line.split("@README", 1)[1].strip()
            result = route_skill.route(query, rules=self.rules, inventory=self.inventory)
            primary = result["primary"]["skill"] if result["primary"] else None
            self.assertEqual(primary, expected, "%r -> %s" % (query, primary))


if __name__ == "__main__":
    unittest.main()
