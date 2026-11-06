"""Fill PyRIT-encoded poison payloads and check that every case retrieves what it should."""
import asyncio
import hashlib
import importlib.metadata as md
import json
import sys
from pathlib import Path

from pyrit.converter import Base64Converter, ROT13Converter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import rag_pipeline as rp  # noqa: E402


async def main():
    src = json.loads((ROOT / "data/rag_suite.json").read_text(encoding="utf-8"))
    payload = src["instruction_payload_for_converters"]
    b64 = (await Base64Converter().convert_async(prompt=payload)).output_text
    rot = (await ROT13Converter().convert_async(prompt=payload)).output_text
    problems = []
    for c in src["cases"]:
        if c.get("poison"):
            c["poison"]["text"] = c["poison"]["text"].replace("__B64__", b64).replace("__ROT13__", rot)
        docs = rp.build_index(c, src["corpus"])
        got = [d["id"] for d in rp.bm25(c["query"], docs)]
        c["retrieved_ids_expected"] = got
        need = list(c["reference_ids"]) + ([c["poison"]["id"]] if c.get("poison") else [])
        missing = [i for i in need if i not in got]
        if missing:
            problems.append((c["id"], missing, got))
    src["pyrit_version"] = md.version("pyrit")
    canon = json.dumps(src, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    src["suite_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    (ROOT / "data/rag_suite_built.json").write_text(json.dumps(src, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in src["cases"]:
        print(c["id"], c["category"], c["retrieved_ids_expected"])
    print("PROBLEMS:", problems)


asyncio.run(main())
