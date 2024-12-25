
import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT


def export_lines(num_lines_in_each_package=100000, num_packages=10):
    cc100_text_file = DATA_SYNTHETIC_ROOT / "ja.txt"

    id_count = 0
    with open(cc100_text_file, 'r', encoding='utf-8') as file:
        for package_count in range(num_packages):
            line_count = 0
            data = []
            for line in tqdm(file, desc=f"creating package {package_count:04} of {num_packages}"):
                id_count += 1
                stripped_line = line.strip()
                # skip too short line
                if len(stripped_line) <= 2:
                    continue

                row = {}
                row["source"] = "cc-100"
                row["id"] = f"cc-100_{id_count}"
                row["line"] = stripped_line

                data.append(row)

                line_count += 1
                if line_count >= num_lines_in_each_package:
                    break

            data = pd.DataFrame(data)
            data.to_csv(DATA_SYNTHETIC_ROOT / "lines" / f"{package_count:04}.csv", index=False)


if __name__ == "__main__":
    export_lines()
