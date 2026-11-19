"""Research 路由回归测试：Supervisor-Skills 的专有路由、CCFA 的既有优先级、显式点名与 GPU 边界。

这些断言针对真实的 route_skill.route() 与仓库清单，不镜像规则表本身。
已有的 CCFA 路由断言仍在 test_skills_router.py 中，本文件只补充 Supervisor 相关的边界。

运行（仓库根目录）:
  python3 -m unittest discover -s tests -p 'test_research_routes.py' -v
"""

import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROUTER_SCRIPTS = os.path.join(REPO, "skills", "routers", "agent-skill-router", "scripts")
sys.path.insert(0, ROUTER_SCRIPTS)
import route_skill  # noqa: E402

SUPERVISOR_PRIMARY = {"idea-evaluator", "deep-research", "intro-drafter", "benchmark-paper-template",
                      "tech-paper-template", "figure-designer", "drawio-reconstruction",
                      "pre-submission-reviewer", "rebuttal-guidance", "vibe-research-workflow",
                      "paper-writer", "paper-polish"}


class ResearchRouteTest(unittest.TestCase):
    def setUp(self):
        self.rules = route_skill.load_rules()
        self.inventory = route_skill.load_inventory()

    def route(self, query, artifacts=()):
        return route_skill.route(query, artifacts, rules=self.rules, inventory=self.inventory)

    def check(self, query, primary, artifacts=(), forbidden_primary=(), no_companions=False):
        result = self.route(query, artifacts)
        got = result["primary"]["skill"] if result["primary"] else None
        self.assertEqual(got, primary, "%r -> %s" % (query, got))
        self.assertNotIn(got, set(forbidden_primary), query)
        if no_companions:
            self.assertEqual(result["companions"], [], query)
        return result

    # --- Supervisor 的专有功能：路由到对应 skill ---

    def test_idea_fatal_flaw_request_routes_to_idea_evaluator(self):
        result = self.check("帮我评估这个研究 idea 的致命缺陷和范式转移风险", "idea-evaluator")
        self.assertEqual(result["primary"]["catalog"], "supervisor-skills")

    def test_survey_grade_investigation_routes_to_deep_research(self):
        self.check("做一个综述级的文献调研，给出 MECE 的证据结构", "deep-research")

    def test_intro_request_routes_to_intro_drafter(self):
        self.check("帮我写引言，按 six-paragraph 的逻辑链组织", "intro-drafter")

    def test_benchmark_paper_request_routes_to_benchmark_template(self):
        self.check("这篇 benchmark 论文的 construction pipeline 应该怎么设计", "benchmark-paper-template")

    def test_reference_image_to_drawio_routes_to_supervisor_drawio(self):
        self.check("把这张参考图重建成可编辑的 draw.io 文件", "drawio-reconstruction")

    def test_drawio_artifact_routes_to_drawio_reconstruction(self):
        self.check("这个图的 xml 有问题，帮我检查", "drawio-reconstruction", artifacts=["fig/arch.drawio"])

    def test_reviewer_mindset_routes_to_rebuttal_guidance_not_ccf_rebuttal(self):
        self.check("帮我分析一下这位审稿人心态，给出 rebuttal 策略", "rebuttal-guidance",
                   forbidden_primary={"ccf-rebuttal-writer", "ccf-paper-reviewer"})

    def test_explicit_supervisor_name_wins_over_ccf_phrases(self):
        self.check("用 $deep-research 技能检索，并顺便帮我写引言", "deep-research",
                   forbidden_primary={"ccf-literature-searcher", "intro-drafter"})

    def test_explicit_deep_research_name_is_not_the_generic_research_skill(self):
        self.check("use deep-research skill to survey GPU scheduling papers", "deep-research",
                   forbidden_primary={"research"})

    # --- CCFA 仍是既有论文工作流的主路由 ---

    def test_generic_literature_retrieval_stays_with_ccf(self):
        self.check("帮我做相关工作的文献检索，找 CUDA 调度的论文", "ccf-literature-searcher",
                   forbidden_primary=SUPERVISOR_PRIMARY)

    def test_paper_review_request_stays_with_ccf(self):
        self.check("帮我审稿这篇论文草稿", "ccf-paper-reviewer", forbidden_primary=SUPERVISOR_PRIMARY)

    def test_rebuttal_writing_stays_with_ccf(self):
        self.check("帮我写 rebuttal 回复审稿人的意见", "ccf-rebuttal-writer",
                   forbidden_primary=SUPERVISOR_PRIMARY)

    def test_explicit_ccf_name_still_wins(self):
        self.check("用 ccf-literature-searcher 检索 CUDA 调度相关论文", "ccf-literature-searcher",
                   forbidden_primary=SUPERVISOR_PRIMARY)

    # --- 非论文领域不受 Supervisor 规则影响 ---

    def check_not_supervisor(self, query):
        result = self.route(query)
        got = result["primary"]["skill"] if result["primary"] else None
        self.assertNotIn(got, SUPERVISOR_PRIMARY, "%r -> %s" % (query, got))

    def test_gpu_kernel_slowness_is_not_research(self):
        result = self.route("这个 kernel 为什么慢，帮我看看 warp stall")
        got = result["primary"]["skill"] if result["primary"] else None
        self.assertNotIn(got, SUPERVISOR_PRIMARY)

    def test_fatal_flaw_in_business_code_is_not_idea_evaluator(self):
        self.check_not_supervisor("请审查这段业务代码里的致命缺陷")

    def test_mece_sales_plan_is_not_deep_research(self):
        self.check_not_supervisor("帮我用 MECE 整理这份销售计划")

    def test_explicit_idea_evaluator_name_wins_in_business_code_context(self):
        self.check("用 idea-evaluator skill 审查这段业务代码的致命缺陷", "idea-evaluator")

    def test_explicit_deep_research_name_wins_in_sales_plan_context(self):
        self.check("用 $deep-research 整理这份销售计划的 MECE 结构", "deep-research")

    def test_mece_over_research_papers_still_routes_to_deep_research(self):
        self.check("用 MECE 梳理这些论文的证据结构", "deep-research")

    def test_vibe_coding_request_is_not_vibe_research(self):
        result = self.route("vibe coding 帮我写一个 CUDA 小工具")
        got = result["primary"]["skill"] if result["primary"] else None
        self.assertNotEqual(got, "vibe-research-workflow")


if __name__ == "__main__":
    unittest.main()
