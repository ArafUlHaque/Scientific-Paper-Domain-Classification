"""Integration checks use the author's real selected artifacts."""
import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest
from demo.artifacts import resolve_artifacts
from demo.inference import BertClassifier, LogisticClassifier, normalize_text, validate_text


class DemoTests(unittest.TestCase):
    def test_input_validation(self):
        for text in [" \n\t", "x" * 20001]:
            with self.assertRaises(ValueError):
                validate_text(text)
        self.assertEqual(normalize_text(" ＡＢＣ\n  Science "), "abc science")

    def test_baseline_out_of_vocabulary(self):
        classifier = LogisticClassifier(resolve_artifacts())
        with self.assertRaisesRegex(ValueError, "no terms recognized"):
            classifier.predict("zzzzzzqqqqqq")

    def test_ui_examples_comparison_and_stale_results(self):
        app = AppTest.from_file("demo/app.py", default_timeout=90).run()
        self.assertFalse(app.exception)
        app.button[1].click().run()
        self.assertTrue(app.error)
        app.selectbox[0].select("Computer Science").run()
        app.button[0].click().run()
        self.assertIn("graph neural network", app.text_area[0].value)
        app.radio[0].set_value("Compare both").run()
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        results = app.session_state["results"]
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(len(result["scores"]), 7)
            self.assertAlmostEqual(sum(result["scores"].values()), 1, places=5)
        app.text_area[0].set_value("An edited abstract").run()
        self.assertNotIn("results", app.session_state)

    def test_bert_truncation(self):
        classifier = BertClassifier(resolve_artifacts())
        result = classifier.predict("Scientific research into computer networks. " * 120)
        self.assertTrue(result.truncated)
        self.assertGreater(result.token_count, 384)


if __name__ == "__main__":
    unittest.main()
