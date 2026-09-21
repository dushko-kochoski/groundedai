from app.services.ingestion.chunking import ParagraphChunker


def test_chunking_is_deterministic_and_combines_small_paragraphs() -> None:
    chunker = ParagraphChunker(target_chars=35, overlap_chars=0)
    text = "First short paragraph.\n\nSecond paragraph.\n\nThird paragraph is longer."

    first = chunker.chunk(text)
    second = chunker.chunk(text)

    assert first == second
    assert [chunk.index for chunk in first] == list(range(len(first)))
    assert first[0].text == "First short paragraph."
    assert all(chunk.text for chunk in first)


def test_overlap_uses_previous_chunk_suffix() -> None:
    chunker = ParagraphChunker(target_chars=25, overlap_chars=10)

    chunks = chunker.chunk("alpha beta gamma delta\n\none two three four five")

    assert len(chunks) == 2
    assert chunks[1].text.startswith("delta\n\n")


def test_large_paragraph_is_split_without_empty_chunks() -> None:
    chunker = ParagraphChunker(target_chars=20, overlap_chars=0)

    chunks = chunker.chunk("word " * 30)

    assert len(chunks) > 1
    assert all(0 < chunk.char_count <= 20 for chunk in chunks)
    assert "" not in [chunk.text for chunk in chunks]


def test_large_unbroken_text_is_split_at_hard_boundary() -> None:
    chunker = ParagraphChunker(target_chars=10, overlap_chars=0)

    chunks = chunker.chunk("x" * 25)

    assert [chunk.char_count for chunk in chunks] == [10, 10, 5]

