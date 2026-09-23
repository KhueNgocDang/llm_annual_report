import unittest

import duckdb

from database import init_db, sync_companies_from_stocks


class CompanySyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.con = duckdb.connect(":memory:")
        init_db(cls.con)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.con.close()

    def setUp(self) -> None:
        self.con.execute("DELETE FROM companies")
        self.con.execute("DELETE FROM stocks")
        self.con.executemany(
            """
            INSERT INTO stocks (
                code, type, floor, status, company_name, company_name_eng,
                short_name, listed_date, delisted_date, company_id, tax_code, isin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "AAA",
                    "stock",
                    "HOSE",
                    "listed",
                    "AAA Corp",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    "BBB",
                    "stock",
                    "HNX",
                    "listed",
                    "BBB Corp",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    "CCC",
                    "stock",
                    "HOSE",
                    "listed",
                    "CCC Corp",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ],
        )

    def test_sync_companies_from_stocks_adds_all_unique_tickers(self) -> None:
        result = sync_companies_from_stocks(self.con)

        companies = self.con.execute(
            "SELECT ticker FROM companies ORDER BY ticker"
        ).fetchall()

        self.assertEqual(result, {"matched": 3, "added": 3, "existing": 0})
        self.assertEqual(companies, [("AAA",), ("BBB",), ("CCC",)])

    def test_sync_companies_from_stocks_can_filter_by_exchange(self) -> None:
        result = sync_companies_from_stocks(self.con, ["HOSE"])

        companies = self.con.execute(
            "SELECT ticker FROM companies ORDER BY ticker"
        ).fetchall()

        self.assertEqual(result, {"matched": 2, "added": 2, "existing": 0})
        self.assertEqual(companies, [("AAA",), ("CCC",)])


if __name__ == "__main__":
    unittest.main()