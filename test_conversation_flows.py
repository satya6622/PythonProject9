"""Six end-to-end conversation scenarios for the interactive matching agent."""
import tempfile
import unittest
from pathlib import Path

from matching_agent import McpClient, ResumeMatchingAgent, extract_requirements


class ConversationFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "R001.txt").write_text("Name: John Doe\nTitle: Frontend Engineer\nSummary: 5+ years experience\nSkills: React, TypeScript, JavaScript, Node.js", encoding="utf-8")
        (root / "R002.txt").write_text("Name: Jane Roe\nTitle: Software Engineer\nSummary: 2+ years experience\nSkills: React, Python, SQL", encoding="utf-8")
        (root / "R003.txt").write_text("Name: Sam Lee\nTitle: Designer\nSummary: 6+ years experience\nSkills: Figma", encoding="utf-8")
        self.client, self.agent = McpClient(self.temp.name), ResumeMatchingAgent(McpClient(self.temp.name))

    def tearDown(self):
        self.client.close()
        self.agent.client.close()
        self.temp.cleanup()

    def test_01_extracts_must_and_nice_requirements(self):
        result = extract_requirements("React and 3+ years required; TypeScript preferred")
        self.assertEqual(result["min_years"], 3)
        self.assertIn("React", result["must_have"])
        self.assertIn("TypeScript", result["nice_to_have"])

    def test_02_finds_candidates(self):
        self.assertIn("John Doe", self.agent.answer("Find candidates with React and 3+ years experience"))

    def test_03_compares_top_candidates(self):
        self.agent.answer("Find candidates with React")
        self.assertIn("Candidate comparison", self.agent.answer("Compare R001 and R002 side by side"))

    def test_04_explains_ranking(self):
        self.agent.answer("Find candidates with React and 3+ years experience")
        self.assertIn("scored", self.agent.answer("Why did R001 rank higher?"))

    def test_05_refines_and_reranks(self):
        self.agent.answer("Find candidates with React")
        self.assertIn("Refined", self.agent.answer("Require TypeScript and at least 3 years"))

    def test_06_generates_screening_questions(self):
        self.agent.answer("Find candidates with React")
        self.assertIn("Walk me through", self.agent.answer("questions for R001"))


if __name__ == "__main__":
    unittest.main()
