from __future__ import annotations

from config import bootstrap_directories
from database import init_db


def main() -> None:
    created = bootstrap_directories()
    db_path = init_db()

    print("Rebuild bootstrap complete")
    print(f"Database: {db_path}")
    print("Directories:")
    for path in created:
        print(f"- {path}")


if __name__ == "__main__":
    main()
