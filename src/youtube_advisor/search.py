from __future__ import annotations
import functools
import numpy as np, pathlib, json, re
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
from .build_embeddings import _parse_md, _chunks

_tok = re.compile(r"\w+")


@functools.lru_cache(maxsize=4)
def _get_model(model_name: str) -> TextEmbedding:
    """Cache TextEmbedding instances; cold-loading the ONNX model is ~1.5s."""
    return TextEmbedding(model_name=model_name)

def _load_corpus_chunks_uncached(corpus_dir: pathlib.Path) -> list[dict]:
    chunks = []
    for md in sorted((corpus_dir / "transcripts").glob("*.md")):
        fm, body = _parse_md(md)
        vid = fm.get("video_id")
        if not vid:
            continue
        for i, ch in enumerate(_chunks(body)):
            chunks.append({
                "video_id": vid,
                "chunk_index": i,
                "text": ch["text"],
                "start_sec": ch["start_sec"],
                "title": fm.get("title", ""),
                "video_url": fm.get("video_url"),
                "published_date": fm.get("published_date"),
            })
    return chunks


def _corpus_mtime(corpus_dir: pathlib.Path) -> float:
    """Max mtime across the transcripts dir + embeddings.npz — used as a
    cache key so a corpus rebuild invalidates the in-memory caches."""
    mts = [p.stat().st_mtime for p in (corpus_dir / "transcripts").glob("*.md")]
    npz = corpus_dir / "references" / "embeddings.npz"
    if npz.exists():
        mts.append(npz.stat().st_mtime)
    return max(mts) if mts else 0.0


@functools.lru_cache(maxsize=4)
def _cached_chunks(corpus_dir_str: str, mtime: float) -> list[dict]:
    return _load_corpus_chunks_uncached(pathlib.Path(corpus_dir_str))


@functools.lru_cache(maxsize=4)
def _cached_npz(corpus_dir_str: str, mtime: float) -> tuple[np.ndarray, np.ndarray, str]:
    npz = np.load(pathlib.Path(corpus_dir_str) / "references" / "embeddings.npz",
                  allow_pickle=False)
    return npz["vectors"].astype(np.float32), npz["offsets"], str(npz["model"])


def _load_corpus_chunks(corpus_dir: pathlib.Path) -> list[dict]:
    """Cached corpus loader. Keyed on (corpus_dir, max_mtime) so a rebuild
    automatically busts the cache; per-query cost drops from ~700ms to <1ms
    on warm calls (matches the existing _get_model lru_cache pattern)."""
    return _cached_chunks(str(corpus_dir), _corpus_mtime(corpus_dir))

def _bm25_rank(chunks: list[dict], q: str, top_k: int) -> list[tuple[int, float]]:
    tokenized = [_tok.findall(c["text"].lower()) for c in chunks]
    bm = BM25Okapi(tokenized)
    scores = bm.get_scores(_tok.findall(q.lower()))
    idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

def _semantic_rank(corpus_dir: pathlib.Path, chunks: list[dict], q: str, top_k: int) -> list[tuple[int, float]]:
    vecs, offsets, model_name = _cached_npz(str(corpus_dir), _corpus_mtime(corpus_dir))
    model = _get_model(model_name)
    qv = np.array(list(model.embed([q]))[0], dtype=np.float32)
    sims = vecs @ qv
    # Map (video_id, chunk_index) → position in chunks[], so we can return
    # indices INTO chunks[] (what _rrf and the caller expect). The npz row
    # order is build-time append order and need not match the sorted-glob
    # order chunks[] uses, so we must join on the offsets key.
    chunk_pos = {(c["video_id"], c["chunk_index"]): i for i, c in enumerate(chunks)}
    order = np.argsort(sims)[::-1]
    out: list[tuple[int, float]] = []
    for row in order:
        vid = str(offsets[row]["video_id"])
        ci = int(offsets[row]["chunk_index"])
        pos = chunk_pos.get((vid, ci))
        if pos is None:
            continue
        out.append((pos, float(sims[row])))
        if len(out) >= top_k:
            break
    return out

def _rrf(rank_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rl in rank_lists:
        for rank, (idx, _) in enumerate(rl):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def search(corpus_dir: pathlib.Path, query: str, mode: str = "hybrid",
           top_k: int = 10) -> list[dict]:
    chunks = _load_corpus_chunks(corpus_dir)
    if not chunks:
        return []
    if mode == "keyword":
        ranked = _bm25_rank(chunks, query, top_k)
    elif mode == "semantic":
        ranked = _semantic_rank(corpus_dir, chunks, query, top_k)
    else:
        ranked = _rrf([_bm25_rank(chunks, query, top_k * 3),
                       _semantic_rank(corpus_dir, chunks, query, top_k * 3)])[:top_k]

    out = []
    for idx, score in ranked:
        c = chunks[idx]
        snippet = c["text"][:300].replace("\n", " ")
        ts_link = f"{c['video_url']}&t={c['start_sec']}" if c["video_url"] else None
        out.append({
            "video_id": c["video_id"],
            "chunk_index": c["chunk_index"],
            "title": c["title"],
            "published_date": c["published_date"],
            "start_sec": c["start_sec"],
            "score": float(score),
            "snippet": snippet,
            "timestamp_link": ts_link,
        })
    return out

def cli():
    import click
    @click.command()
    @click.argument("query", nargs=-1, required=True)
    @click.option("--corpus", type=click.Path(exists=True), default=".")
    @click.option("--mode", type=click.Choice(["keyword", "semantic", "hybrid"]), default="hybrid")
    @click.option("--top-k", type=int, default=10)
    def _main(query, corpus, mode, top_k):
        res = search(pathlib.Path(corpus), " ".join(query), mode=mode, top_k=top_k)
        click.echo(json.dumps(res, ensure_ascii=False, indent=2))
    _main()

if __name__ == "__main__":
    cli()
