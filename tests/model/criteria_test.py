from unittest import TestCase, main

from avalan.model.criteria import KeywordStoppingCriteria

class FakeTokenizer:
    def decode(self, token_id, skip_special_tokens=False):
        return token_id

class KeywordStoppingCriteriaTestCase(TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_single_keyword(self):
        criteria = KeywordStoppingCriteria(["END"], self.tokenizer)
        self.assertFalse(criteria([["E"]], None))
        self.assertFalse(criteria([["N"]], None))
        self.assertTrue(criteria([["D"]], None))

    def test_multiple_keywords_any(self):
        criteria = KeywordStoppingCriteria(["Hello", "world"], self.tokenizer)
        self.assertFalse(criteria([["Hel"]], None))
        self.assertFalse(criteria([["lo "]], None))
        self.assertTrue(criteria([["world"]], None))

    def test_multiple_keywords_all_must_be_present(self):
        criteria = KeywordStoppingCriteria(["foo", "bar"], self.tokenizer, all_must_be_present=True)
        self.assertFalse(criteria([["foo "]], None))
        self.assertTrue(criteria([["bar"]], None))

    def test_no_keywords_assertion(self):
        with self.assertRaises(AssertionError):
            KeywordStoppingCriteria([], self.tokenizer)

if __name__ == "__main__":
    main()
