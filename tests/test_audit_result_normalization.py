import unittest

from llm_financial_statement_audit import _normalize_audit_result


class AuditResultNormalizationTests(unittest.TestCase):
    def test_recovers_firm_and_opinion_from_details_when_top_level_fields_are_empty(self) -> None:
        result = {
            "found": False,
            "external_audit_firm": None,
            "external_audit_firm_en": None,
            "audit_opinion": None,
            "signing_auditor_names": [],
            "details": [
                {
                    "type": "firm",
                    "name": "Deloitte Vietnam",
                    "evidence": "Công ty kiểm toán Deloitte Việt Nam",
                },
                {
                    "type": "opinion",
                    "opinion": "chấp nhận toàn phần",
                },
            ],
            "reason": "",
        }

        normalized = _normalize_audit_result(result, "Báo cáo kiểm toán độc lập ...")

        self.assertTrue(normalized["found"])
        self.assertEqual(normalized["external_audit_firm"], "Deloitte Vietnam")
        self.assertEqual(normalized["audit_opinion"], "unqualified")

    def test_maps_opinion_phrase_from_source_text_when_llm_does_not_fill_top_level_field(self) -> None:
        result = {
            "found": False,
            "external_audit_firm": None,
            "external_audit_firm_en": None,
            "audit_opinion": None,
            "signing_auditor_names": [],
            "details": [],
            "reason": "",
        }

        normalized = _normalize_audit_result(
            result,
            "Ý kiến của kiểm toán viên: chấp nhận toàn phần về báo cáo tài chính",
        )

        self.assertTrue(normalized["found"])
        self.assertEqual(normalized["audit_opinion"], "unqualified")


if __name__ == "__main__":
    unittest.main()
