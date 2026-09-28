"""Run with python3 -m unittest test_graph_scoring.py."""

import math
import random
import unittest

from graph_scoring import official_metrics, validate_predictions


def example_rows(fake, presence=None):
    presence = presence or [1] * len(fake)
    labels = []
    predictions = []
    for index, (label, present) in enumerate(zip(fake, presence, strict=True)):
        labels.append({
            "ID": str(index), "FILE_FAKE": label, "VOICE_FAKE": label,
            "MUSIC_FAKE": label, "VOICE_PRESENT": present, "MUSIC_PRESENT": present,
        })
        predictions.append({
            "ID": str(index), "FILE_FAKE_PROB": 0.5, "VOICE_FAKE_PROB": 0.5,
            "MUSIC_FAKE_PROB": 0.5, "VOICE_PRESENT_PROB": 0.5,
            "MUSIC_PRESENT_PROB": 0.5,
        })
    return labels, predictions


class OfficialScoringTests(unittest.TestCase):
    def test_tied_scores_use_discrete_official_eer_without_interpolation(self):
        # Given: ROC points (0,0), (0,0.5), (1,1), with a three-way score tie.
        labels, predictions = example_rows([1, 0, 1, 0], [1, 0, 1, 0])
        for row, score in zip(predictions, [0.9, 0.8, 0.8, 0.8], strict=True):
            row["FILE_FAKE_PROB"] = score
            row["VOICE_PRESENT_PROB"] = score
        # When: applying the published nearest-ROC-point rule.
        report = official_metrics(labels, predictions)
        # Then: EER is 0.25; interpolation would incorrectly yield 1/3.
        self.assertEqual(report["metrics"]["file_eer"], 0.25)
        self.assertEqual(report["metrics"]["voice_presence_auc"], 0.75)

    def test_component_eer_uses_true_presence_even_when_prediction_disagrees(self):
        # Given: absent components would corrupt an otherwise perfect ranking.
        labels, predictions = example_rows([0, 0, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0])
        for row, score in zip(predictions, [0.1, 0.2, 0.8, 0.9, 0.99, 0.01], strict=True):
            row["VOICE_FAKE_PROB"] = score
            row["MUSIC_FAKE_PROB"] = score
            row["VOICE_PRESENT_PROB"] = 0.0
        # When: scoring component predictions.
        report = official_metrics(labels, predictions)
        # Then: only the four truly present components determine their EER.
        self.assertEqual(report["metrics"]["voice_eer"], 0.0)
        self.assertEqual(report["metrics"]["music_eer"], 0.0)
        self.assertEqual(report["counts"]["voice_eer"], 4)
        self.assertAlmostEqual(report["score"], 0.725)

    def test_unknown_labels_and_single_class_metrics_do_not_fabricate_total(self):
        # Given: one unknown file label, complete component labels, no music absence.
        labels, predictions = example_rows([0, 1, 0, 1], [1, 1, 0, 0])
        labels[0]["FILE_FAKE"] = None
        labels[2]["VOICE_FAKE"] = ""
        for row in labels:
            row["MUSIC_PRESENT"] = 1
        # When: reporting the available evidence.
        report = official_metrics(labels, predictions)
        # Then: unknown absent voice truth is irrelevant; incomplete totals stay null.
        self.assertIsNone(report["score"])
        self.assertEqual(report["metrics"]["file_eer"], 0.5)
        self.assertIsNone(report["metrics"]["music_presence_auc"])
        self.assertEqual(report["metrics"]["voice_eer"], 0.5)
        self.assertIn("file_eer", report["partial"])
        self.assertEqual(report["coverage"]["file_eer"], 0.75)
        self.assertEqual(report["unknown_counts"]["file_eer"], 1)
        self.assertIn("music_presence_auc", report["invalid"])

    def test_unknown_presence_keeps_subset_metrics_but_invalidates_total(self):
        # Given: an unknown presence label prevents deciding component eligibility.
        labels, predictions = example_rows([0, 1, 0, 1], [1, 1, 0, 0])
        labels[3]["VOICE_PRESENT"] = None
        # When: computing component and presence metrics.
        report = official_metrics(labels, predictions)
        # Then: neither metric may quietly drop that uncertainty.
        self.assertEqual(report["metrics"]["voice_eer"], 0.5)
        self.assertEqual(report["metrics"]["voice_presence_auc"], 0.5)
        self.assertIn("voice_eer", report["partial"])
        self.assertIn("voice_presence_auc", report["partial"])
        self.assertIsNone(report["score"])

    def test_discrete_roc_matches_official_sklearn_expression_with_ties(self):
        # Given: imbalanced cohorts and deliberately tied predictions.
        try:
            import numpy as np
            from sklearn.metrics import roc_auc_score, roc_curve
        except ImportError:
            self.skipTest("Optional sklearn differential check needs numpy and scikit-learn")
        rng = random.Random(236749)
        for size in range(3, 45):
            targets = [0, 1] + [rng.randrange(2) for _ in range(size - 2)]
            scores = [rng.randrange(8) / 7 for _ in targets]
            labels, predictions = example_rows(targets, targets)
            for row, score in zip(predictions, scores, strict=True):
                row["FILE_FAKE_PROB"] = score
                row["VOICE_PRESENT_PROB"] = score
            fpr, tpr, _ = roc_curve(targets, scores, pos_label=1, drop_intermediate=False)
            fnr = 1 - tpr
            index = np.argmin(np.abs(fpr - fnr))
            # When: comparing the stdlib implementation against the actual library.
            report = official_metrics(labels, predictions)
            # Then: threshold selection is exact, including floating-point tie breaks.
            self.assertEqual(report["metrics"]["file_eer"], (fpr[index] + fnr[index]) / 2)
            self.assertAlmostEqual(report["metrics"]["voice_presence_auc"], roc_auc_score(targets, scores))

    def test_rejects_corrupted_submission_values_ids_and_columns(self):
        # Given: realistic CSV corruption, including order drift and missing scores.
        _, valid = example_rows([0, 1])
        variants = []
        for value in [math.nan, math.inf, -0.1, 1.1]:
            damaged = [dict(row) for row in valid]
            damaged[0]["FILE_FAKE_PROB"] = value
            variants.append(damaged)
        variants.extend([valid[::-1], [valid[0], valid[0]]])
        missing = [dict(row) for row in valid]
        del missing[0]["VOICE_FAKE_PROB"]
        variants.append(missing)
        extra = [dict(row) for row in valid]
        extra[0]["EXTRA"] = 0.5
        variants.append(extra)
        # When / Then: the submission boundary rejects each defect before export.
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_predictions(rows, ["0", "1"])


if __name__ == "__main__":
    unittest.main()
