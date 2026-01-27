import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - surface error on use
    faiss = None
    _FAISS_IMPORT_ERROR = exc

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - surface error on use
    SentenceTransformer = None
    _ST_IMPORT_ERROR = exc

DATA_PATH = Path("data/Pokemon.csv")
STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "faiss.index"
META_PATH = STORAGE_DIR / "metadata.json"


STAT_ALIASES = {
    "hp": "HP",
    "attack": "Attack",
    "atk": "Attack",
    "defense": "Defense",
    "def": "Defense",
    "sp atk": "Sp. Atk",
    "sp. atk": "Sp. Atk",
    "sp attack": "Sp. Atk",
    "special attack": "Sp. Atk",
    "sp def": "Sp. Def",
    "sp. def": "Sp. Def",
    "sp defense": "Sp. Def",
    "special defense": "Sp. Def",
    "speed": "Speed",
    "total": "Total",
    "generation": "Generation",
    "gen": "Generation",
}

OP_WORDS = [
    (">=", [">=", "at least", "minimum", "min", "no less than"]),
    ("<=", ["<=", "at most", "maximum", "max", "no more than"]),
    (">", [">", "greater than", "more than", "over"]),
    ("<", ["<", "less than", "under", "below"]),
]


@dataclass
class FilterSpec:
    column: str
    op: str
    value: object


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for local embeddings."
            ) from _ST_IMPORT_ERROR
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")


class PokemonRAG:
    def __init__(self) -> None:
        self.df = self._load_dataframe()
        self.known_types = self._collect_types(self.df)
        self.embedder = self._load_embedder()
        self.records, self.texts = self._load_or_build_index()

    def _load_dataframe(self) -> pd.DataFrame:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        return df

    def _collect_types(self, df: pd.DataFrame) -> List[str]:
        type_1 = df["Type 1"].dropna().unique().tolist()
        type_2 = df["Type 2"].dropna().unique().tolist()
        known = sorted({t for t in type_1 + type_2 if str(t).strip()})
        return known

    def _load_embedder(self) -> LocalEmbedder:
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return LocalEmbedder(model_name)

    def _index_is_stale(self) -> bool:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            return True
        data_mtime = DATA_PATH.stat().st_mtime
        return data_mtime > INDEX_PATH.stat().st_mtime or data_mtime > META_PATH.stat().st_mtime

    def _load_or_build_index(self) -> Tuple[List[Dict[str, object]], List[str]]:
        if self._index_is_stale():
            return self._build_index()

        with META_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        records = [item["record"] for item in metadata]
        texts = [item["text"] for item in metadata]

        if len(records) != len(texts):
            return self._build_index()

        if faiss is None:
            raise RuntimeError("faiss is required to load the vector index.") from _FAISS_IMPORT_ERROR

        if not INDEX_PATH.exists():
            return self._build_index()

        self.index = faiss.read_index(str(INDEX_PATH))
        return records, texts

    def _build_index(self) -> Tuple[List[Dict[str, object]], List[str]]:
        records, texts = self._build_documents(self.df)
        embeddings = self.embedder.embed(texts)

        if faiss is None:
            raise RuntimeError("faiss is required to build the vector index.") from _FAISS_IMPORT_ERROR

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_PATH))

        metadata = [{"record": record, "text": text} for record, text in zip(records, texts)]
        with META_PATH.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        self.index = index
        return records, texts

    def _build_documents(self, df: pd.DataFrame) -> Tuple[List[Dict[str, object]], List[str]]:
        records: List[Dict[str, object]] = []
        texts: List[str] = []

        for _, row in df.iterrows():
            record = self._row_to_record(row)
            text = self._record_to_text(record)
            records.append(record)
            texts.append(text)

        return records, texts

    def _row_to_record(self, row: pd.Series) -> Dict[str, object]:
        def _to_int(value: object) -> int:
            try:
                return int(value)
            except Exception:
                return 0

        def _to_bool(value: object) -> bool:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() == "true"

        return {
            "id": _to_int(row.get("#")),
            "name": str(row.get("Name", "")).strip(),
            "type_1": str(row.get("Type 1", "")).strip(),
            "type_2": str(row.get("Type 2", "")).strip(),
            "total": _to_int(row.get("Total")),
            "hp": _to_int(row.get("HP")),
            "attack": _to_int(row.get("Attack")),
            "defense": _to_int(row.get("Defense")),
            "sp_atk": _to_int(row.get("Sp. Atk")),
            "sp_def": _to_int(row.get("Sp. Def")),
            "speed": _to_int(row.get("Speed")),
            "generation": _to_int(row.get("Generation")),
            "legendary": _to_bool(row.get("Legendary")),
        }

    def _record_to_text(self, record: Dict[str, object]) -> str:
        type_2 = record.get("type_2") or "None"
        return (
            f"Pokemon {record.get('name')} (#{record.get('id')}) | "
            f"Types: {record.get('type_1')} / {type_2} | "
            f"Stats - HP {record.get('hp')}, Attack {record.get('attack')}, Defense {record.get('defense')}, "
            f"Sp. Atk {record.get('sp_atk')}, Sp. Def {record.get('sp_def')}, Speed {record.get('speed')}, "
            f"Total {record.get('total')} | "
            f"Generation {record.get('generation')} | Legendary {record.get('legendary')}"
        )

    def build_context(self, records: List[Dict[str, object]]) -> str:
        if not records:
            return ""
        lines = []
        for idx, record in enumerate(records, start=1):
            lines.append(f"{idx}. {self._record_to_text(record)}")
        return "\n".join(lines)

    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        filters = self._parse_filters(query)
        if filters:
            filtered = self._apply_filters(self.df, filters)
            if filtered.empty:
                return [], {"strategy": "filter", "filters": filters}
            records = [self._row_to_record(row) for _, row in filtered.iterrows()]
            records = sorted(records, key=lambda r: r.get("name", ""))
            return records[:top_k], {"strategy": "filter", "filters": filters, "total_matches": len(records)}

        query_embedding = self.embedder.embed([query])
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Dict[str, object]] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.records):
                continue
            results.append(self.records[idx])

        return results, {"strategy": "vector", "filters": []}

    def _parse_filters(self, query: str) -> List[FilterSpec]:
        q = query.lower()
        filters: List[FilterSpec] = []
        seen_columns = set()

        if "legendary" in q:
            if any(token in q for token in ["not legendary", "non legendary", "non-legendary", "not a legendary"]):
                filters.append(FilterSpec("Legendary", "=", False))
            else:
                filters.append(FilterSpec("Legendary", "=", True))

        gen_match = re.search(r"(?:generation|gen)\s*(\d+)", q)
        if gen_match:
            filters.append(FilterSpec("Generation", "=", int(gen_match.group(1))))
            seen_columns.add("Generation")

        type_filters = []
        for type_name in self.known_types:
            if re.search(rf"\b{re.escape(type_name.lower())}\b", q):
                type_filters.append(type_name)

        if type_filters:
            filters.append(FilterSpec("Type", "in", type_filters))

        for alias, column in STAT_ALIASES.items():
            if column in seen_columns:
                continue
            alias_pattern = re.escape(alias)
            pattern_a = rf"(?P<stat>{alias_pattern})\s*(?P<op>>=|<=|=|>|<)?\s*(?P<value>\d+)"
            pattern_b = rf"(?P<value>\d+)\s*(?P<stat>{alias_pattern})"

            match = re.search(pattern_a, q)
            if not match:
                match = re.search(pattern_b, q)

            if not match:
                continue

            groups = match.groupdict()
            op = groups.get("op")
            value = int(groups.get("value") or 0)

            if op is None:
                window_start = max(match.start() - 24, 0)
                context = q[window_start : match.end() + 24]
                op = self._infer_op(context)

            filters.append(FilterSpec(column, op, value))
            seen_columns.add(column)

        return filters

    def _infer_op(self, fragment: str) -> str:
        for op, words in OP_WORDS:
            if any(word in fragment for word in words):
                return op
        return "="

    def _apply_filters(self, df: pd.DataFrame, filters: List[FilterSpec]) -> pd.DataFrame:
        filtered = df.copy()
        for filt in filters:
            column = filt.column
            op = filt.op
            value = filt.value

            if column == "Legendary":
                filtered = filtered[filtered["Legendary"].astype(str).str.lower() == str(value).lower()]
                continue

            if column == "Type":
                type_values = set(value)
                mask = filtered["Type 1"].isin(type_values) | filtered["Type 2"].isin(type_values)
                filtered = filtered[mask]
                continue

            if column not in filtered.columns:
                continue

            series = pd.to_numeric(filtered[column], errors="coerce")
            if op == ">=":
                filtered = filtered[series >= int(value)]
            elif op == "<=":
                filtered = filtered[series <= int(value)]
            elif op == ">":
                filtered = filtered[series > int(value)]
            elif op == "<":
                filtered = filtered[series < int(value)]
            else:
                filtered = filtered[series == int(value)]

        return filtered

    def format_results(self, records: List[Dict[str, object]]) -> str:
        if not records:
            return "No matching Pokemon found."
        lines = []
        for record in records:
            lines.append(
                f"- {record['name']} (HP {record['hp']}, Attack {record['attack']}, Defense {record['defense']}, "
                f"Sp. Atk {record['sp_atk']}, Sp. Def {record['sp_def']}, Speed {record['speed']}, "
                f"Total {record['total']}, Gen {record['generation']}, Legendary {record['legendary']})"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    rag = PokemonRAG()
    print(f"Index ready with {len(rag.records)} entries.")
