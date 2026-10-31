from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SkillStructureTest(unittest.TestCase):
    def test_core_skill_stays_concise(self):
        lines = (ROOT / "SKILL.md").read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(lines), 250)

    def test_local_references_from_skill_exist(self):
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        targets = re.findall(r"\[[^]]+]\((references/[^)]+\.md)\)", skill)
        self.assertTrue(targets)
        for target in targets:
            with self.subTest(target=target):
                self.assertTrue((ROOT / target).is_file())

    def test_relative_links_resolve(self):
        # Covers asset links from SKILL.md and cross-links between references,
        # resolved relative to the file that contains them.
        sources = [ROOT / "SKILL.md", *sorted((ROOT / "references").glob("*.md"))]
        checked = 0
        for source in sources:
            text = source.read_text(encoding="utf-8")
            prose = re.sub(r"^```.*?^```[ \t]*$", "", text, flags=re.MULTILINE | re.DOTALL)
            for target in re.findall(r"\[[^]]+]\(([^)\s]+)\)", prose):
                if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                path = target.split("#", 1)[0]
                with self.subTest(source=source.name, target=target):
                    self.assertTrue((source.parent / path).is_file())
                checked += 1
        self.assertGreater(checked, 0)

    def test_controlled_writing_has_four_groups_and_twelve_examples(self):
        reference = (
            ROOT / "references" / "controlled-technical-chinese.md"
        ).read_text(encoding="utf-8")
        for heading in ("操作手册", "API 文档", "故障排查", "产品介绍"):
            with self.subTest(heading=heading):
                self.assertIn(f"### {heading}", reference)
        self.assertEqual(len(re.findall(r"^#### 样例 \d+$", reference, re.MULTILINE)), 12)

    def test_paragraph_wrapping_rule_is_documented(self):
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        reference = (
            ROOT / "references" / "terminology-and-typography.md"
        ).read_text(encoding="utf-8")
        self.assertIn("一段一行", skill)
        self.assertIn("## 段落换行与源文件行宽", reference)
        self.assertIn("scripts/unwrap_md_paragraphs.py", reference)

    # ---- GPU / AI Infra 定制部分 ----

    def test_gpu_references_are_linked_from_skill(self):
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        for name in (
            "gpu-report-structures.md",
            "evidence-and-numbers.md",
            "gpu-terminology.md",
            "figures-formulas-code.md",
            "output-formats.md",
        ):
            with self.subTest(name=name):
                self.assertIn(f"(references/{name})", skill)
                self.assertTrue((ROOT / "references" / name).is_file())

    def test_evidence_levels_are_consistent(self):
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        evidence = (ROOT / "references" / "evidence-and-numbers.md").read_text(encoding="utf-8")
        for level in ("示意", "设计", "引用", "源码", "估算", "编译产物", "实测"):
            with self.subTest(level=level):
                self.assertIn(f"| {level} |", skill)
                self.assertIn(f"| {level} |", evidence)

    def test_report_structures_cover_seven_types(self):
        reference = (ROOT / "references" / "gpu-report-structures.md").read_text(encoding="utf-8")
        prose = re.sub(r"^```.*?^```[ \t]*$", "", reference, flags=re.MULTILINE | re.DOTALL)
        self.assertEqual(len(re.findall(r"^## \d\. ", prose, re.MULTILINE)), 7)

    def test_gpu_regression_examples(self):
        reference = (ROOT / "references" / "evidence-and-numbers.md").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"^#### 样例 G\d+$", reference, re.MULTILINE)), 12)

    def test_html_template_is_self_contained(self):
        template = (ROOT / "assets" / "report-template.html").read_text(encoding="utf-8")
        self.assertIn("color-scheme: light", template)
        self.assertIn("badge-supplement", template)
        self.assertIsNone(re.search(r"""(?:src|href)\s*=\s*["']https?://""", template))
        self.assertNotIn("@import", template)
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("(assets/report-template.html)", skill)

    def test_project_override_is_explicitly_a_template(self):
        reference = (
            ROOT / "references" / "project-overrides-example.md"
        ).read_text(encoding="utf-8")
        self.assertIn("本文件只是模板", reference)
        self.assertFalse((ROOT / "references" / "Project-Overrides.md").exists())


if __name__ == "__main__":
    unittest.main()
