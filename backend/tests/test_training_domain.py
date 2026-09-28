import unittest

from app.training.domain import determine_learner_result


class LearnerResultTests(unittest.TestCase):
    def test_no_selection_is_not_answered(self):
        result = determine_learner_result(["A"], [])

        self.assertEqual(result, "not_answered")

    def test_exact_selection_is_correct(self):
        result = determine_learner_result(["A", "C"], ["C", "A"])

        self.assertEqual(result, "correct")

    def test_disjoint_selection_is_incorrect(self):
        result = determine_learner_result(["A"], ["B"])

        self.assertEqual(result, "incorrect")

    def test_incomplete_selection_is_partially_correct(self):
        result = determine_learner_result(["A", "C"], ["A"])

        self.assertEqual(result, "partially_correct")

    def test_extra_wrong_selection_is_partially_correct(self):
        result = determine_learner_result(["A"], ["A", "B"])

        self.assertEqual(result, "partially_correct")


if __name__ == "__main__":
    unittest.main()