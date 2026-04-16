"""
Разбиение Markdown-документов на чанки для RAG.

Идея «семантичности» без отдельной модели эмбеддингов:
- сначала границы по заголовкам (разделы);
- внутри раздела — склейка по параграфам до целевого размера (одна-две мысли на чанк);
- один огромный параграф режется по предложениям, затем по символам;
- небольшое перекрытие между соседними чанками для контекста при поиске.

Опционально: лимит в токенах (tiktoken), если нужно под конкретный embedder.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Heading-based split
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _parse_heading(line: str) -> Optional[Tuple[int, str]]:
    m = _HEADING_RE.match(line.rstrip())
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


def split_markdown_by_headings(text: str) -> List[Tuple[Tuple[str, ...], str]]:
    """
    Разбить markdown на пары (путь_заголовков, тело).

    Текст до первого заголовка идёт с путём ().
    """
    lines = text.splitlines()
    stack: List[Tuple[int, str]] = []
    buf: List[str] = []
    out: List[Tuple[Tuple[str, ...], str]] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        path = tuple(t for _, t in stack)
        out.append((path, body))

    for line in lines:
        h = _parse_heading(line)
        if h is not None:
            flush()
            level, title = h
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return out


# ---------------------------------------------------------------------------
# Semantic-ish body split: paragraphs first, then sentences, then hard cut
# ---------------------------------------------------------------------------

_PAR_BREAK = re.compile(r"\n\s*\n+")


def _split_sentences_fallback(text: str) -> List[str]:
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\d«\"(])", text)
    return [p.strip() for p in parts if p.strip()]


def _merge_up_to_limit(parts: List[str], max_piece: int, separator: str) -> List[str]:
    if not parts:
        return []
    merged: List[str] = []
    cur = parts[0]
    for p in parts[1:]:
        if len(cur) + len(separator) + len(p) <= max_piece:
            cur = cur + separator + p
        else:
            merged.append(cur)
            cur = p
    merged.append(cur)
    return merged


def _hard_char_windows(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    step = max(1, max_chars - overlap_chars)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start += step
    return [c for c in chunks if c]


def _apply_overlap(pieces: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    if overlap_chars <= 0 or len(pieces) <= 1:
        return pieces
    out: List[str] = [pieces[0]]
    for i in range(1, len(pieces)):
        prev = out[-1]
        suf = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        cand = (suf + "\n\n" + pieces[i]).strip() if suf else pieces[i]
        if len(cand) <= max_chars:
            out.append(cand)
        else:
            out.append(pieces[i])
    return out


def semantic_split_body(
    text: str,
    max_chars: int,
    overlap_chars: int = 0,
    *,
    min_merge_chars: int = 120,
) -> List[str]:
    """
    Склеивает параграфы в чанки до max_chars (типично 500–1000 символов для плотного RAG).

    Короткие «обломки» (< min_merge_chars) по возможности приклеиваются к соседнему параграфу
    внутри того же окна, чтобы не плодить чанки из одной строки.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paras = [p.strip() for p in _PAR_BREAK.split(text) if p.strip()]
    if not paras:
        return _hard_char_windows(text, max_chars, overlap_chars)

    packed: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush_buf() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        packed.append("\n\n".join(buf))
        buf, buf_len = [], 0

    for p in paras:
        if len(p) > max_chars:
            flush_buf()
            packed.extend(_split_oversized_paragraph(p, max_chars, overlap_chars))
            continue

        sep = 2 if buf else 0
        add = len(p) + sep
        if buf and buf_len + add > max_chars:
            flush_buf()

        if not buf:
            buf = [p]
            buf_len = len(p)
            continue

        if buf_len + add <= max_chars:
            buf.append(p)
            buf_len += add
        else:
            flush_buf()
            buf = [p]
            buf_len = len(p)

    flush_buf()

    # склеить слишком короткие хвосты с соседями
    if min_merge_chars > 0 and len(packed) > 1:
        merged: List[str] = []
        i = 0
        while i < len(packed):
            cur = packed[i]
            if len(cur) < min_merge_chars and i + 1 < len(packed):
                nxt = packed[i + 1]
                if len(cur) + 2 + len(nxt) <= max_chars:
                    merged.append(cur + "\n\n" + nxt)
                    i += 2
                    continue
            merged.append(cur)
            i += 1
        packed = merged

    if overlap_chars > 0:
        packed = _apply_overlap(packed, max_chars, overlap_chars)
    return packed


def _split_oversized_paragraph(p: str, max_chars: int, overlap_chars: int) -> List[str]:
    sents = _split_sentences_fallback(p)
    if len(sents) > 1:
        pieces = _merge_up_to_limit(sents, max_chars, " ")
        if all(len(x) <= max_chars for x in pieces):
            out = _apply_overlap(pieces, max_chars, overlap_chars) if overlap_chars else pieces
            return out
    return _hard_char_windows(p, max_chars, overlap_chars)


def _try_count_tokens(text: str, encoding_name: str = "cl100k_base") -> Optional[int]:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return None


def split_text_to_token_chunks(
    text: str,
    max_tokens: int,
    encoding_name: str,
    overlap_chars: int,
) -> List[str]:
    """Разбить один фрагмент так, чтобы каждая часть была ≤ max_tokens (tiktoken)."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        approx = max(200, int(max_tokens * 3.2))
        return semantic_split_body(text, approx, overlap_chars, min_merge_chars=0)

    def ntok(s: str) -> int:
        return len(enc.encode(s))

    if ntok(text) <= max_tokens:
        return [text]

    approx = max(200, int(max_tokens * 3.2))
    rough = semantic_split_body(text, approx, 0, min_merge_chars=0)
    out: List[str] = []
    buf = ""
    for piece in rough:
        if not buf:
            buf = piece
            continue
        trial = buf + "\n\n" + piece
        if ntok(trial) <= max_tokens:
            buf = trial
        else:
            out.extend(_shrink_chunk_to_tokens(buf, max_tokens, enc))
            buf = piece
    if buf:
        out.extend(_shrink_chunk_to_tokens(buf, max_tokens, enc))

    # Перекрытие по символам здесь не добавляем: оно ломает бюджет токенов без повторного сплита.
    return out


def _shrink_chunk_to_tokens(chunk: str, max_tokens: int, enc: Any) -> List[str]:
    """Бинарный поиск по длине строки в символах, пока куски не укладываются в бюджет токенов."""
    result: List[str] = []
    rest = chunk.strip()
    while rest:
        if len(enc.encode(rest)) <= max_tokens:
            result.append(rest)
            break
        lo, hi = 0, len(rest)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(enc.encode(rest[:mid])) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        take = lo if lo > 0 else 1
        piece = rest[:take].strip()
        if piece:
            result.append(piece)
        rest = rest[take:].strip()
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    heading_path: Tuple[str, ...]
    chunk_index: int
    source_path: str = ""
    char_len: int = 0
    token_len: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["heading_path"] = list(self.heading_path)
        return d


def chunk_document(
    text: str,
    *,
    source_path: str = "",
    max_chars: int = 900,
    overlap_chars: int = 100,
    max_tokens: Optional[int] = None,
    tiktoken_encoding: str = "cl100k_base",
    min_merge_chars: int = 120,
) -> List[Chunk]:
    """
    Заголовки задают раздел -> внутри раздела параграфная склейка до max_chars.

    По умолчанию max_chars около 900 (~200–280 токенов), удобно для точного поиска и узких ответов.
    """
    sections = split_markdown_by_headings(text)
    chunks: List[Chunk] = []
    idx = 0

    for path, body in sections:
        bodies = semantic_split_body(
            body,
            max_chars,
            overlap_chars,
            min_merge_chars=min_merge_chars,
        )
        for b in bodies:
            if max_tokens is not None:
                parts = split_text_to_token_chunks(b, max_tokens, tiktoken_encoding, overlap_chars)
            else:
                parts = [b]
            for part in parts:
                ntok = _try_count_tokens(part, tiktoken_encoding)
                chunks.append(
                    Chunk(
                        text=part,
                        heading_path=path,
                        chunk_index=idx,
                        source_path=source_path,
                        char_len=len(part),
                        token_len=ntok,
                    )
                )
                idx += 1

    return chunks


def iter_docs_from_directory(
    root: Path,
    patterns: Tuple[str, ...] = ("*.md", "*.markdown"),
) -> Iterator[Tuple[Path, str]]:
    root = root.resolve()
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            try:
                yield p, p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Markdown documents into RAG chunks.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Файл .md или каталог с документами",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=900,
        help="Целевой верхний предел размера чанка (символы), по умолчанию ~одна мысль",
    )
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=None, help="Жёсткий лимит токенов (нужен tiktoken)")
    parser.add_argument("--min-merge-chars", type=int, default=120, help="Ниже этого длины клеить с соседом")
    parser.add_argument("--jsonl", type=str, default=None, help="Записать результат в JSONL")
    args = parser.parse_args()

    if args.path is None:
        here = Path(__file__).resolve().parent.parent
        docs = here / "docs"
        if docs.is_dir():
            paths = list(iter_docs_from_directory(docs))
        else:
            print("Укажите путь к .md файлу или положите документы в ./docs")
            return
    else:
        p = Path(args.path)
        if p.is_file():
            paths = [(p, p.read_text(encoding="utf-8", errors="replace"))]
        elif p.is_dir():
            paths = list(iter_docs_from_directory(p))
        else:
            print("Путь не найден:", args.path)
            return

    all_rows: List[Dict[str, Any]] = []
    for fp, content in paths:
        rel = str(fp)
        chunks = chunk_document(
            content,
            source_path=rel,
            max_chars=args.max_chars,
            overlap_chars=args.overlap,
            max_tokens=args.max_tokens,
            min_merge_chars=args.min_merge_chars,
        )
        print(f"{rel}: {len(chunks)} chunks")
        for c in chunks:
            all_rows.append(c.to_dict())

    if args.jsonl:
        out = Path(args.jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("Wrote", out)


if __name__ == "__main__":
    main()
