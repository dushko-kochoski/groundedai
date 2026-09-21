import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


class ParagraphChunker:
    def __init__(self, target_chars: int, overlap_chars: int) -> None:
        if target_chars < 1:
            raise ValueError("target_chars must be positive")
        if overlap_chars < 0 or overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than target_chars")
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> list[TextChunk]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        units = [piece for paragraph in paragraphs for piece in self._split_paragraph(paragraph)]
        base_chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        for unit in units:
            separator_size = 2 if current else 0
            if current and current_size + separator_size + len(unit) > self.target_chars:
                base_chunks.append("\n\n".join(current))
                current = []
                current_size = 0
                separator_size = 0
            current.append(unit)
            current_size += separator_size + len(unit)
        if current:
            base_chunks.append("\n\n".join(current))

        chunks: list[TextChunk] = []
        for index, base_chunk in enumerate(base_chunks):
            if index == 0 or self.overlap_chars == 0:
                chunk_text = base_chunk
            else:
                overlap = self._overlap_suffix(base_chunks[index - 1])
                chunk_text = f"{overlap}\n\n{base_chunk}" if overlap else base_chunk
            if chunk_text:
                chunks.append(TextChunk(index=len(chunks), text=chunk_text))
        return chunks

    def _split_paragraph(self, paragraph: str) -> list[str]:
        pieces: list[str] = []
        remaining = paragraph.strip()
        while len(remaining) > self.target_chars:
            split_at = remaining.rfind(" ", 0, self.target_chars + 1)
            if split_at <= 0:
                split_at = self.target_chars
            piece = remaining[:split_at].strip()
            if piece:
                pieces.append(piece)
            remaining = remaining[split_at:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    def _overlap_suffix(self, text: str) -> str:
        if len(text) <= self.overlap_chars:
            return text
        suffix = text[-self.overlap_chars :]
        first_space = suffix.find(" ")
        if first_space >= 0:
            suffix = suffix[first_space + 1 :]
        return suffix.strip()

