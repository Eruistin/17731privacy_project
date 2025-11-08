import json
from pathlib import Path
import random

def _read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    SEED = 33
    path = Path(f"./wiki_json/train_finetune_{SEED}.json")
    texts = _read_json(path)
    random.seed(SEED)
    random.shuffle(texts)
    sample_nums = len(texts)

    train_num = sample_nums // 2
    _write_json(Path(f"./wiki_json/train_shadow_{SEED}.json"), texts[:train_num])
    _write_json(Path(f"./wiki_json/test_shadow_{SEED}.json"), texts[train_num:])

    print(f"Train num is {train_num}, test num is {len(texts) - train_num}")


if __name__ == "__main__":
    main()