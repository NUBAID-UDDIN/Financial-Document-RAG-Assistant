import unittest

from evaluation import answer_matches, is_abstention, retrieval_matches, validate_cases


class DummyDoc:
    def __init__(self, text):
        self.page_content = text


class EvaluationTests(unittest.TestCase):
    def test_text_answer_matches_punctuation_insensitively(self):
        case = {"id": "vendor", "answerable": True, "accepted_answers": ["Komal Chawla"], "match_mode": "text"}
        self.assertTrue(answer_matches("Answer: KOMAL CHAWLA\nSource: vendor KOMAL CHAWLA", case))

    def test_wrong_number_in_source_quote_does_not_pass(self):
        case = {"id": "amount", "answerable": True, "accepted_answers": ["241.00"], "match_mode": "number"}
        self.assertFalse(answer_matches("Answer: 1241.00\nSource: Total due 241.00", case))
        self.assertTrue(answer_matches("Answer: ₹241.00\nSource: Total due 241.00", case))

    def test_unanswerable_requires_abstention(self):
        case = {"id": "unknown", "answerable": False, "accepted_answers": []}
        self.assertTrue(answer_matches("Not found in document.", case))
        self.assertFalse(answer_matches("The capital is Paris.", case))
        self.assertTrue(is_abstention("The answer is not specified in the document."))

    def test_validation_rejects_unreplaced_ground_truth_placeholder(self):
        cases = [{
            "id": "amount", "question": "What is the amount?", "answerable": True,
            "accepted_answers": ["COPY_REAL_AMOUNT_FROM_PDF"],
            "evidence_terms": ["COPY_REAL_AMOUNT_FROM_PDF"], "match_mode": "number"
        }]
        with self.assertRaises(ValueError):
            validate_cases(cases)

    def test_retrieval_hit_checks_annotated_evidence(self):
        case = {"id": "invoice", "answerable": True, "evidence_terms": ["INV-123"]}
        retrieved = [{"doc": DummyDoc("Invoice Number: INV-123")}]
        self.assertTrue(retrieval_matches(retrieved, case))
        self.assertFalse(retrieval_matches([{"doc": DummyDoc("Invoice Number: INV-124")}], case))


if __name__ == "__main__":
    unittest.main()
