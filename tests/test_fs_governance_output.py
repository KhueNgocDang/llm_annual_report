import unittest

from llm_governance import (
    _get_governance_result_item_code,
    _normalize_financial_statement_governance_result,
)


class FinancialStatementGovernanceOutputTests(unittest.TestCase):
    def test_uses_suffixed_item_code_for_financial_statement_governance_storage(self) -> None:
        self.assertEqual(
            _get_governance_result_item_code(
                "GOV_DIRECTORY",
                "financial_statement_document_embeddings",
            ),
            "GOV_DIRECTORY_FS",
        )

    def test_normalizes_financial_statement_roster_details_to_name_and_gender(self) -> None:
        result = {
            "found": True,
            "value": {"total_members": 2, "women_count": 1, "men_count": 1},
            "details": [
                {"name": "Nguyễn Thị A", "position": "Thành viên", "gender": "female"},
                {"name": "Trần Văn B", "position": "Chủ tịch", "gender": "male"},
            ],
            "reason": "",
        }

        normalized = _normalize_financial_statement_governance_result(
            "GOV_DIRECTORY_FS",
            result,
            [],
        )

        self.assertEqual(
            normalized["details"],
            [
                {"name": "Nguyễn Thị A", "gender": "Bà"},
                {"name": "Trần Văn B", "gender": "Ông"},
            ],
        )
        self.assertEqual(normalized["value"], {})


if __name__ == "__main__":
    unittest.main()
