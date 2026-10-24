"""agent-skill-router 的路由行为、清单完整性与清单生成测试。

运行（仓库根目录）:
  python3 -m unittest discover -s tests -p 'test_skills_router.py' -v

测试分四类:
  1. route_skill 行为：显式点名（覆盖清单全部 skill）、制品 vs 关键词、实现 vs 性能、论文阅读 vs 通用写作、
     审稿 vs 代码审查、微信授权门槛、REA 与 sepia、catalog 回退的置信度。
  2. 清单与真实 checkout 的一致性：四个请求来源的 pin 与 skill 来源；规则引用的 skill 必须可安装或声明 requires。
  3. availability 标签：source / skill_copy / runtime 三者分别判断，不混用。
  4. build_inventory 使用临时 fixture 树：同名先到先得、保留名、根目录 SKILL.md 取 frontmatter 名、
     逗号分隔目录、runtime-skills 缺失、--check 过期检测。
"""

import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROUTER_DIR = os.path.join(REPO, "skills", "routers", "agent-skill-router")
SCRIPTS = os.path.join(ROUTER_DIR, "scripts")
REFERENCES = os.path.join(ROUTER_DIR, "references")
INVENTORY = os.path.join(REFERENCES, "inventory.tsv")
CATCHALL = os.path.join(REPO, "skills", "routers", "gpu-development-catchall", "SKILL.md")
ROUTER_SKILL_FILES = [
    os.path.join(REPO, "skills", "routers", name, "SKILL.md")
    for name in (
        "gpu-development-catchall",
        "gpu-kernel-authoring-router",
        "gpu-performance-router",
        "gpu-framework-router",
        "llm-serving-router",
        "agent-skill-router",
    )
]

sys.path.insert(0, SCRIPTS)
import build_inventory  # noqa: E402
import route_skill  # noqa: E402

# 路由器 SKILL.md 中出现、但当前 checkout 不一定可安装的名称：
#   nsys-profile-analysis / ncu-profile-analysis 来自可选的 VeloQ（install-veloq.sh）；
#   claude-delegate / codex-background 为用户级 symlink，不由本仓库安装。
OPTIONAL_EXTERNAL = {"nsys-profile-analysis", "ncu-profile-analysis", "claude-delegate", "codex-background"}
# 反引号中形如 skill 名、但实际是 kind 取值、仓库名或 route_skill 输出标签的词。
# 标签只在 SKILL.md 的“Availability labels”一节列出；test_output_label_exemptions_are_real_labels
# 确认这些标签确实由 route_skill.py 输出，因此豁免不会掩盖真正缺失的 skill 引用。
OUTPUT_LABELS = {"not-checked", "not-found", "on-path", "user-level", "runtime-cli", "missing-source",
                 "agent-gpu-skills"}
NOT_SKILL_TOKENS = OUTPUT_LABELS

REQUESTED_SOURCES = {"sepia", "wx-cli", "rea", "mattpocock-skills"}


def _load_inventory_rows():
    rows = []
    with open(INVENTORY, encoding="utf-8") as f:
        columns = None
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if columns is None:
                columns = fields
                continue
            rows.append(dict(zip(columns, fields)))
    return rows


def _installed_names():
    names = {}
    for row in _load_inventory_rows():
        if row["kind"] == "skill" and row["status"] == "installed":
            names.setdefault(row["name"], row["catalog"])
    return names


class RouteBehaviorTest(unittest.TestCase):
    rules = route_skill.load_rules()
    inventory = route_skill.load_inventory()

    def route(self, query, artifacts=(), skill_roots=None):
        return route_skill.route(query, artifacts, rules=self.rules, inventory=self.inventory,
                                 skill_roots=skill_roots)

    def assertPrimary(self, result, skill, status="installable"):
        self.assertIsNotNone(result["primary"], "no primary for %r" % skill)
        self.assertEqual(result["primary"]["skill"], skill)
        self.assertEqual(result["primary"]["status"], status)

    def test_profile_question_without_artifact_asks_for_evidence(self):
        result = self.route("帮我看下这个 kernel 为什么慢")
        self.assertPrimary(result, "gpu-performance-router")
        self.assertTrue(any("profiling" in n for n in result["notes"]))

    def test_profile_artifact_outranks_keywords(self):
        result = self.route("这个 kernel 为什么慢，帮我分析", ["prof/run.ncu-rep"])
        self.assertPrimary(result, "ncu-report-skill")

    def test_implementation_names_tool_not_performance_router(self):
        self.assertPrimary(self.route("写一个 Triton kernel 做 fused add"), "triton-skill")

    def test_nsys_timeline_is_performance_not_implementation(self):
        self.assertPrimary(self.route("nsys 时间线里 launch 间隔很大"), "perf-nsight-systems")

    def test_tilelang_compile_error_goes_to_debugging(self):
        self.assertPrimary(self.route("tilelang kernel 编译报错，debug 一下"), "debugging-tilelang-programs")

    def test_explicit_rule_skill_request_wins(self):
        result = self.route("用 tdd skill 给这个函数补测试")
        self.assertPrimary(result, "tdd")
        self.assertEqual(result["companions"], [])

    def test_explicit_request_covers_inventory_skill_without_rule(self):
        # grill-me 没有路由规则，只靠清单中的名称即可被点名（来自 mattpocock-skills）。
        result = self.route("用 $grill-me 拷问一下这个方案")
        self.assertPrimary(result, "grill-me")
        self.assertEqual(result["primary"]["domain"], "mattpocock-skills")

    def test_explicit_inventory_skill_beats_earlier_plain_mention(self):
        result = self.route("先看 review 的结果，然后用 grill-me skill 继续")
        self.assertPrimary(result, "grill-me")

    def test_plain_word_review_is_not_explicit_skill_request(self):
        result = self.route("review 这个 diff 的代码改动")
        self.assertIsNotNone(result["primary"])
        self.assertNotIn("explicit:review", result["primary"]["why"])

    def test_software_code_review_not_paper_review(self):
        result = self.route("帮我 review 这个 PR 的代码改动")
        self.assertIn(result["primary"]["skill"], {"ponytail-review", "review", "code-review",
                                                    "high-performance-code-review"})

    def test_paper_review_not_software_review(self):
        self.assertPrimary(self.route("帮我 review 论文初稿"), "ccf-paper-reviewer")

    def test_rebuttal_is_paper_workflow(self):
        self.assertPrimary(self.route("帮我写 rebuttal 回复审稿人的意见"), "ccf-rebuttal-writer")

    def test_chinese_technical_report_is_writing_skill_not_paper_writer(self):
        self.assertPrimary(self.route("帮我润色这段中文技术报告，是 kernel 原理报告"), "tech-doc-style-chinese")

    def test_paper_reading_with_pdf_artifact_uses_pdf_skill(self):
        self.assertPrimary(self.route("读这篇论文并总结方法", ["papers/attention.pdf"]), "pdf")

    def test_zotero_library_request_routes_to_zotero(self):
        result = self.route("Zotero 文库里找 attention 相关的文献")
        self.assertPrimary(result, "zotero-cli")
        self.assertEqual(result["primary"]["availability"]["runtime_bin"], "zotero-cli")

    def test_de_ai_writing_routes_to_sepia(self):
        result = self.route("帮我把这段文章去掉 AI 味，像人写的")
        self.assertPrimary(result, "sepia")
        self.assertIn("../sepia/SKILL.md", result["primary"]["requires"])

    def inventory_without(self, name):
        inventory = dict(self.inventory)
        inventory.pop(name, None)
        return inventory

    def test_reverse_engineering_routes_to_rea_skill_with_runtime_requirement(self):
        # 清单中没有 REA skill（runtime-skills 尚未链接）时：标记为 absent，并给出 requires。
        result = route_skill.route("帮我逆向分析这个 Electron 应用的 asar 包", rules=self.rules,
                                   inventory=self.inventory_without("reverse-engineer-anything"))
        self.assertPrimary(result, "reverse-engineer-anything", status="absent")
        self.assertIn("rea-agents@6.7.0", result["primary"]["requires"])
        self.assertEqual(result["primary"]["availability"]["runtime_bin"], "rea")

    def test_reverse_engineering_is_installable_once_runtime_skills_exist(self):
        inventory = dict(self.inventory)
        inventory["reverse-engineer-anything"] = {"status": "installed", "catalog": "rea",
                                                  "source": "repos/rea/runtime-skills/reverse-engineer-anything",
                                                  "description": "Reverse engineer."}
        result = route_skill.route("帮我逆向分析这个 Electron 应用的 asar 包", rules=self.rules,
                                   inventory=inventory)
        self.assertPrimary(result, "reverse-engineer-anything", status="installable")

    def test_wechat_request_routes_with_direct_authorization_gate(self):
        result = self.route("帮我导出微信群里昨天的聊天记录，我已经授权")
        self.assertPrimary(result, "wx-cli")
        gate = result["primary"]["gate"]
        self.assertIn("无需再次确认", gate)
        self.assertIn("不运行 sudo wx init", gate)
        self.assertEqual(result["primary"]["availability"]["runtime_bin"], "wx")

    def test_explicit_uninstalled_skill_is_not_substituted(self):
        result = route_skill.route("用 reverse-engineer-anything skill 帮我整理一下这个包", rules=self.rules,
                                   inventory=self.inventory_without("reverse-engineer-anything"))
        self.assertPrimary(result, "reverse-engineer-anything", status="absent")
        self.assertEqual(result["companions"], [])
        self.assertIn("不会替换", " ".join(result["notes"]))

    def test_user_level_delegate_skill_is_never_installable(self):
        result = self.route("用 codex-background skill 跑这个任务")
        self.assertPrimary(result, "codex-background", status="user-level")

    def test_no_evidence_returns_no_primary(self):
        self.assertIsNone(self.route("hello there")["primary"])

    def test_unmatched_planning_query_does_not_hit_wechat_on_one_bigram(self):
        self.assertIsNone(self.route("规划一下这次数据库迁移的排期和风险清单")["primary"])

    def test_catalog_fallback_is_labelled_low_confidence(self):
        skills = {"alpha-planner": {"status": "installed", "catalog": "fixture", "source": "x",
                                    "description": "Plan database migration schedules and risk lists."}}
        result = route_skill.route("plan the database migration schedule", rules=[], inventory=skills)
        self.assertTrue(result["catalog_fallback"])
        self.assertEqual(result["primary"]["skill"], "alpha-planner")
        self.assertIn("低置信度", " ".join(result["notes"]))

    def test_at_most_two_companions_and_one_question(self):
        queries = [
            ("kernel 慢，用 nsys 和 ncu 都看看", ["a.nsys-rep", "b.ncu-rep"]),
            ("写 cuda kernel 的 tdd 测试和 code review", []),
            ("sglang serving 的 KV cache 吞吐 benchmark", []),
        ]
        for query, artifacts in queries:
            result = self.route(query, artifacts)
            self.assertLessEqual(len(result["companions"]), 2, query)
            if result["ask"] is not None:
                self.assertEqual(result["ask"].count("？"), 1, query)

    def test_availability_separates_source_copy_and_runtime(self):
        root = tempfile.mkdtemp(prefix="skill-copy-")
        self.addCleanup(shutil.rmtree, root, True)
        os.makedirs(os.path.join(root, "zotero-cli"))
        with open(os.path.join(root, "zotero-cli", "SKILL.md"), "w", encoding="utf-8") as f:
            f.write("---\nname: zotero-cli\n---\n")
        result = self.route("Zotero 文库里找文献", skill_roots=[root])
        avail = result["primary"]["availability"]
        self.assertEqual(avail["source"], "installable")
        self.assertEqual(avail["skill_copy"], "found")
        self.assertIn(avail["runtime"], {"on-path", "missing"})

    def test_missing_copy_is_reported_not_guessed(self):
        root = tempfile.mkdtemp(prefix="empty-skills-")
        self.addCleanup(shutil.rmtree, root, True)
        result = self.route("Zotero 文库里找文献", skill_roots=[root])
        self.assertEqual(result["primary"]["availability"]["skill_copy"], "not-found")


TILELANG_SKILLS = {"writing-tilelang-kernels", "debugging-tilelang-programs", "tilelang-optimization-loop",
                   "optimizing-tilelang-programs", "profiling-tilelang-programs",
                   "torch-profiling-tilelang-programs", "testing-fwd-bwd-kernels",
                   "writing-tilelang-design-docs", "designing-with-tilefoundry"}


class RoutingRegressionMatrixTest(unittest.TestCase):
    """回归矩阵：原始误路由、相邻领域的误捕获，以及显式点名与强制制品的优先级。

    每一行断言的是结果：主 skill（或 None）、不应出现的主 skill / 伴随 skill、是否产生澄清问题。
    """

    rules = route_skill.load_rules()
    inventory = route_skill.load_inventory()

    def route(self, query, artifacts=()):
        return route_skill.route(query, artifacts, rules=self.rules, inventory=self.inventory)

    def check(self, query, primary, artifacts=(), forbidden_primary=(), forbidden_companions=(),
              no_companions=False, no_question=False):
        result = self.route(query, artifacts)
        got = result["primary"]["skill"] if result["primary"] else None
        self.assertEqual(got, primary, "%r -> %s" % (query, got))
        if result["primary"]:
            self.assertNotIn(got, set(forbidden_primary), query)
        companions = {c["skill"] for c in result["companions"]}
        if no_companions:
            self.assertEqual(companions, set(), "%r companions %s" % (query, companions))
        self.assertFalse(companions & set(forbidden_companions), "%r companions %s" % (query, companions))
        if no_question:
            self.assertIsNone(result["ask"], "%r asked %s" % (query, result["ask"]))
        return result

    # --- 原始误路由（Codex 复核中的失败）---

    def test_cuda_literature_search_is_literature_not_scheduling(self):
        self.check("帮我做相关工作的文献检索，找 CUDA 调度的论文", "ccf-literature-searcher",
                   forbidden_primary=TILELANG_SKILLS | {"persistent-kernel-scheduling"},
                   no_companions=True, no_question=True)

    def test_natural_literature_retrieval_holdout_with_generic_retrieval_negative_control(self):
        # 正例：自然语言的论文检索，主题词是 GPU 调度，不应带入伴随 skill。
        self.check("帮我检索关于GPU调度的相关论文", "ccf-literature-searcher",
                   forbidden_primary=TILELANG_SKILLS | {"persistent-kernel-scheduling"},
                   no_companions=True, no_question=True)
        # 负例：没有论文或文献词的通用检索请求，不得路由到文献检索。
        self.check("帮我检索一下订单系统的数据库索引", None,
                   forbidden_primary={"ccf-literature-searcher"})

    def test_repository_audit_is_ponytail_audit_not_paper_review(self):
        self.check("审计一下这个仓库的风险代码和慢路径", "ponytail-audit",
                   forbidden_primary={"ccf-paper-reviewer"}, no_question=True)

    def test_deepstream_video_pipeline_is_deepstream_not_tilelang(self):
        self.check("用 deepstream 搭一个视频分析 pipeline", "deepstream-generate-pipeline",
                   forbidden_primary=TILELANG_SKILLS, forbidden_companions=TILELANG_SKILLS)

    def test_doca_dma_names_the_specialist_skill(self):
        self.check("在 BlueField DPU 上配置 DOCA DMA", "doca-dma", no_companions=True)

    def test_evo2_request_names_the_bionemo_specialist(self):
        self.check("用 Evo 2 NIM 生成一段 DNA 序列", "bionemo-evo2-nim")

    def test_earth2_forecast_names_the_earth2_specialist(self):
        self.check("用 earth2studio 跑一个天气预报 forecast", "earth2studio-deterministic-forecast",
                   forbidden_companions={"earth2studio-create-prognostic"})

    def test_cuopt_milp_names_the_cuopt_specialist(self):
        self.check("用 cuOpt 求解一个 MILP 混合整数规划", "cuopt-numerical-optimization-api")

    # --- 相邻领域：不应被通用动作词或共享领域词捕获 ---

    def test_triton_numerical_debugging_stays_on_triton(self):
        self.check("帮我调试这个 Triton kernel 的数值误差", "triton-skill",
                   forbidden_primary=TILELANG_SKILLS, forbidden_companions=TILELANG_SKILLS)

    def test_generic_service_pipeline_design_doc_has_no_gpu_or_writing_capture(self):
        self.check("帮我设计一个订单服务的数据 pipeline，写设计文档，并做 layout 和 baseline 对比", None,
                   forbidden_primary=TILELANG_SKILLS | {"tech-doc-style-chinese", "tao-run-automl-deft-pipeline"},
                   forbidden_companions=TILELANG_SKILLS)

    def test_generic_cpu_thread_scheduling_is_not_gpu_scheduling(self):
        self.check("帮我设计一个 CPU 线程池的任务调度策略", None,
                   forbidden_primary={"persistent-kernel-scheduling"})

    def test_generic_words_alone_never_select_a_tilelang_skill(self):
        for query in ("帮我 debug 一下这个 pipeline 的 layout", "做一个 baseline 对比的 profiling 计划",
                      "写个设计文档，记录 pipeline 的 grad 流程"):
            self.check(query, None, forbidden_primary=TILELANG_SKILLS, forbidden_companions=TILELANG_SKILLS)

    def test_gpu_report_writing_is_tech_doc_but_generic_design_doc_is_not(self):
        self.check("帮我润色这段中文技术报告，是 kernel 原理报告", "tech-doc-style-chinese", no_question=True)
        result = self.route("帮我写一份订单服务的设计文档")
        primary = result["primary"]["skill"] if result["primary"] else None
        self.assertNotEqual(primary, "tech-doc-style-chinese")

    # --- 显式点名与强制制品优先于门控 ---

    def test_explicit_tilelang_skill_without_tilelang_words_is_honoured(self):
        self.check("用 $writing-tilelang-kernels 帮我看看这个 sample", "writing-tilelang-kernels",
                   no_companions=True)

    def test_explicit_deepstream_dev_overrides_generate_pipeline(self):
        self.check("用 $deepstream-dev 帮我写一个 pyservicemaker 程序", "deepstream-dev")

    def test_actual_ncu_artifact_wins_even_with_generic_words(self):
        self.check("这个 kernel 为什么慢，帮我分析一下 pipeline 和 layout 的问题", "ncu-report-skill",
                   artifacts=["prof/run.ncu-rep"], forbidden_primary=TILELANG_SKILLS,
                   forbidden_companions=TILELANG_SKILLS)

    # --- 其他已验证的边界（不应回退） ---

    def test_chinese_paper_review_is_not_software_review(self):
        self.check("帮我 review 论文初稿", "ccf-paper-reviewer", forbidden_primary={"ponytail-review"})

    def test_positive_control_persistent_kernel_scheduling_still_routes(self):
        self.check("persistent kernel 的 stream-k 调度怎么选", "persistent-kernel-scheduling")

    def test_positive_control_tilelang_layout_with_tilelang_term_still_routes(self):
        self.check("tilelang 的 layout 怎么优化", "optimizing-tilelang-programs")

    def test_zotero_request_routes_to_zotero_cli(self):
        self.check("我的 Zotero 文库里找 attention 的文献", "zotero-cli", no_companions=True)

    def test_explicit_unknown_skill_is_not_substituted(self):
        # 点名的名称不在清单中：不得把它替换为其他 skill。
        result = self.route("用 $no-such-skill-here 帮我做事")
        self.assertIsNone(result["primary"])


class RuleAndInventoryIntegrityTest(unittest.TestCase):
    def test_rules_without_requires_are_installed_in_inventory(self):
        installed = _installed_names()
        for rule in route_skill.load_rules():
            if rule.get("requires"):
                continue
            self.assertIn(rule["skill"], installed, "rule %s has no requires and is not installed" % rule["skill"])

    def test_rule_kinds_and_runtime_bins_are_consistent(self):
        for rule in route_skill.load_rules():
            self.assertIn(rule.get("kind", "instruction"), {"instruction", "runtime-cli", "user-level"}, rule["skill"])
            if rule.get("kind") == "runtime-cli" and rule["skill"] in ("wx-cli", "rea", "zotero-cli"):
                self.assertTrue(rule.get("runtime_bin"), rule["skill"])

    def test_router_skill_names_resolve_or_are_optional(self):
        installed = _installed_names()
        missing = set()
        for path in ROUTER_SKILL_FILES:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for name in re.findall(r"`([a-z][a-z0-9]*(?:-[a-z0-9]+)+)`", text):
                if name in NOT_SKILL_TOKENS:
                    continue
                if name not in installed and name not in OPTIONAL_EXTERNAL:
                    missing.add((os.path.basename(os.path.dirname(path)), name))
        self.assertEqual(sorted(missing), [], "router references skills that are not installed")

    def test_output_label_exemptions_are_real_labels(self):
        source = ""
        for path in (os.path.join(SCRIPTS, "route_skill.py"), os.path.join(SCRIPTS, "build_inventory.py"),
                     os.path.join(REFERENCES, "routing-rules.json")):
            with open(path, encoding="utf-8") as f:
                source += f.read()
        for label in OUTPUT_LABELS - {"agent-gpu-skills"}:
            self.assertIn('"%s"' % label, source, "exempted token %s is not emitted by the router" % label)

    def test_protected_delegate_names_are_not_installed_by_catalogs(self):
        installed = _installed_names()
        self.assertNotIn("claude-delegate", installed)
        self.assertNotIn("codex-background", installed)

    def test_four_requested_sources_are_inventoried(self):
        catalogs = {r["catalog"]: r["status"] for r in _load_inventory_rows() if r["kind"] == "catalog"}
        for source in REQUESTED_SOURCES:
            self.assertIn(source, catalogs)
        self.assertEqual(catalogs["sepia"], "present")
        self.assertEqual(catalogs["wx-cli"], "present")
        self.assertEqual(catalogs["mattpocock-skills"], "present")
        # rea 的 skill 来自 npm 包链接出的 runtime-skills，只有运行安装器后才存在。
        rea_present = os.path.isdir(os.path.join(REPO, "repos", "rea", "runtime-skills"))
        self.assertEqual(catalogs["rea"], "present" if rea_present else "absent")

    def test_sepia_siblings_and_wechat_root_skill_are_named_from_frontmatter(self):
        names = _installed_names()
        for name in ("sepia", "sepia-write", "sepia-review", "sepia-refactor", "sepia-recreate", "sepia-hemingway",
                     "wx-cli"):
            self.assertEqual(names.get(name), "sepia" if name.startswith("sepia") else "wx-cli", name)

    def test_mattpocock_names_that_overlap_cursor_are_shadowed_not_overwritten(self):
        rows = {(r["catalog"], r["name"]): r for r in _load_inventory_rows() if r["kind"] == "skill"}
        for name in ("tdd", "handoff", "prototype", "grill-with-docs"):
            self.assertEqual(rows[("mattpocock-skills", name)]["status"], "shadowed", name)
            self.assertEqual(rows[("mattpocock-skills", name)]["shadowed_by"], "cursor-skills", name)
            self.assertEqual(rows[("cursor-skills", name)]["status"], "installed", name)

    def test_broad_catalog_coverage_is_present(self):
        catalogs = {r["catalog"]: r["status"] for r in _load_inventory_rows() if r["kind"] == "catalog"}
        for catalog in ("local", "nvidia-skills", "cursor-skills", "amd-skills", "ccfa-skills",
                        "kernel-design-agents"):
            self.assertEqual(catalogs.get(catalog), "present", catalog)

    def test_catchall_no_longer_has_generic_non_domain_triggers(self):
        with open(CATCHALL, encoding="utf-8") as f:
            text = f.read()
        for generic in ("推荐", "讲一下", "解释下", "分析下", "帮忙看一下", "有没有资料", "怎么写", "如何实现"):
            self.assertNotIn('"%s"' % generic, text, generic)
        self.assertIn("agent-skill-router", text)

    def test_committed_inventory_matches_checkout(self):
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "build_inventory.py"), "--check"],
            cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


class InventoryGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="inventory-fixture-")
        self.addCleanup(shutil.rmtree, self.root, True)
        self._write_fixture()

    def _write(self, rel, text):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _skill(self, rel, name, description):
        self._write(os.path.join(rel, "SKILL.md"),
                    "---\nname: %s\ndescription: %s\n---\n\n# %s\n" % (name, description, name))

    def _write_fixture(self):
        self._write("install.sh", (
            "SKILL_NAMES=(\n    local-one\n    tdd\n)\n"
            "SKILL_DIRS=(\n    skills/other/local-one\n    skills/other/tdd\n)\n"
        ))
        self._skill("skills/other/local-one", "local-one", "A local skill.")
        self._skill("skills/other/tdd", "tdd", "Local TDD skill.")
        self._skill("repos/cursor-skills/skills/tdd", "tdd", "Cursor copy.")
        self._skill("repos/cursor-skills/skills/grill", "grill", "Cursor grill.")
        self._skill("repos/cursor-skills/skills/claude-delegate", "claude-delegate", "Must not install.")
        self._write("sources/external-sources.conf", (
            "# fixture\n"
            "upstream|https://example.invalid/upstream|UNPINNED|skills|skills/engineering,skills/misc|none|fixture\n"
            "pending-src|https://example.invalid/pending|UNPINNED|pending|-|pending|fixture\n"
            "root-src|https://example.invalid/root|UNPINNED|skills|.|none|fixture\n"
            "rea-like|https://example.invalid/rea|UNPINNED|skills|runtime-skills|npm:fake-pkg@1.0.0:fake-bin|fixture\n"
        ))
        self._skill("repos/upstream/skills/engineering/tdd", "tdd", "Upstream TDD.")
        self._skill("repos/upstream/skills/engineering/fresh-skill", "fresh-skill", "Only in upstream.")
        self._skill("repos/upstream/skills/misc/dir-name", "frontmatter-name", "Name comes from frontmatter.")
        self._skill("repos/upstream/skills/in-progress/hidden", "hidden", "Not configured.")
        # 根目录 SKILL.md：目录基名为 "root-src"，frontmatter 名为 "root-skill"。
        self._skill("repos/root-src", "root-skill", "Root skill from frontmatter.")

    def _rows(self):
        path = build_inventory.inventory_path(self.root)
        with open(path, encoding="utf-8") as f:
            return [l.rstrip("\n").split("\t") for l in f if l.strip() and not l.startswith("#")][1:]

    def _generate(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(build_inventory.main(["--root", self.root]), 0)

    def test_first_claim_wins_and_protected_names_are_marked(self):
        self._generate()
        by_key = {(r[1], r[2]): r for r in self._rows() if r[0] == "skill"}
        self.assertEqual(by_key[("local", "tdd")][3], "installed")
        self.assertEqual(by_key[("cursor-skills", "tdd")][3], "shadowed")
        self.assertEqual(by_key[("upstream", "tdd")][3], "shadowed")
        self.assertEqual(by_key[("upstream", "fresh-skill")][3], "installed")
        self.assertEqual(by_key[("cursor-skills", "claude-delegate")][3], "protected")
        self.assertEqual(by_key[("cursor-skills", "grill")][3], "installed")

    def test_comma_separated_dirs_and_excluded_directories(self):
        self._generate()
        names = {r[2] for r in self._rows() if r[0] == "skill" and r[1] == "upstream"}
        self.assertIn("frontmatter-name", names)
        self.assertNotIn("hidden", names)

    def test_root_skill_takes_frontmatter_name_not_directory_basename(self):
        self._generate()
        by_key = {(r[1], r[2]): r for r in self._rows() if r[0] == "skill"}
        self.assertEqual(by_key[("root-src", "root-skill")][3], "installed")
        self.assertNotIn(("root-src", "root-src"), by_key)

    def test_missing_runtime_skills_root_is_absent(self):
        self._generate()
        catalogs = {r[1]: r[3] for r in self._rows() if r[0] == "catalog"}
        self.assertEqual(catalogs["rea-like"], "absent")
        self.assertEqual(catalogs["pending-src"], "absent")

    def test_generation_is_deterministic_and_check_detects_drift(self):
        self._generate()
        with open(build_inventory.inventory_path(self.root), encoding="utf-8") as f:
            first = f.read()
        self._generate()
        with open(build_inventory.inventory_path(self.root), encoding="utf-8") as f:
            second = f.read()
        self.assertEqual(first, second)

        with redirect_stdout(io.StringIO()):
            self.assertEqual(build_inventory.main(["--root", self.root, "--check"]), 0)

        self._skill("repos/cursor-skills/skills/new-cursor", "new-cursor", "Added after generation.")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(build_inventory.main(["--root", self.root, "--check"]), 1)

    def test_description_is_single_line_and_bounded(self):
        self._skill("skills/other/long", "long", "word " * 200)
        self._write("install.sh", (
            "SKILL_NAMES=(\n    local-one\n    tdd\n    long\n)\n"
            "SKILL_DIRS=(\n    skills/other/local-one\n    skills/other/tdd\n    skills/other/long\n)\n"
        ))
        self._generate()
        for row in self._rows():
            self.assertLessEqual(len(row[6]), build_inventory.DESC_LIMIT)
            self.assertNotIn("\t", row[6])
            self.assertEqual(row[6], row[6].rstrip())


if __name__ == "__main__":
    unittest.main()
