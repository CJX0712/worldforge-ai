"""默认分块器：标题感知的递归字符分块。

作者：晨星
"""

from __future__ import annotations

from worldforge.types import Chunk, Document

_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ")


def _is_heading(line: str) -> bool:
    return line.lstrip().startswith("#")


class RecursiveCharChunker:
    """按标题层级继承路径的递归字符分块器。

    不变量
        I1  返回分块序号从 0 连续递增
        I2  任一分块的 heading_path 等于该分块正文**起始处**所处的标题栈，
            而不是 flush 时刻的标题栈（否则首块会继承到下一章标题）
        I3  标题行自身单独成块，正文不因被并入标题段落而丢失
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self._size = chunk_size
        self._overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces: list[tuple[str, str]] = []
        heading_stack: list[str] = []
        buffer: list[str] = []
        buffer_path: tuple[str, ...] = ()

        def flush() -> None:
            nonlocal buffer
            body = "".join(buffer).strip()
            buffer = []
            if not body:
                return
            for piece in self._split(body):
                pieces.append((piece, "/".join(buffer_path)))

        for line in doc.text.splitlines():
            if _is_heading(line):
                # 先把旧缓冲落盘再更新标题栈，保证 buffer_path 是正文起始处的快照
                flush()
                level = len(line) - len(line.lstrip("#"))
                title = line.lstrip("#").strip()
                heading_stack = heading_stack[: level - 1]
                while len(heading_stack) < level - 1:
                    heading_stack.append("")
                heading_stack.append(title)
                pieces.append((line.strip(), "/".join(heading_stack)))
                continue
            if not buffer:
                buffer_path = tuple(heading_stack)
            buffer.append(line)
            buffer.append("\n")
        flush()

        chunks: list[Chunk] = []
        for ordinal, (text, path) in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{ordinal}",
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    text=text,
                    meta={"heading_path": path, "source": "recursive", **dict(doc.meta)},
                )
            )
        return chunks

    def _split(self, text: str) -> list[str]:
        """在标题层级内做带重叠的递归切分。"""

        out: list[str] = []
        stack = [text]
        while stack:
            current = stack.pop(0)
            if len(current) <= self._size:
                if current.strip():
                    out.append(current.strip())
                continue
            head, tail = self._cut(current)
            # 无进展就必须硬切：carry 回灌可能让 tail 不短于 current，
            # 死循环的表现不是变慢而是直接吃光内存
            if len(tail) >= len(current):
                head, tail = current[: self._size], current[self._size - self._overlap :]
            if len(head) > self._size:
                # 段落整体比窗口还长：先切段自己，再处理余下部分
                stack.insert(0, tail)
                stack.insert(0, head)
                continue
            if head.strip():
                out.append(head.strip())
            if tail.strip():
                stack.insert(0, tail)
        return out

    def _cut(self, text: str) -> tuple[str, str]:
        """在第一个命中的分隔符处切开，并把已消费部分的尾部回灌成重叠。

        carry 必须在 head 长于 overlap 时才回灌：否则 carry 等于整段 head，
        tail 长度不缩短，切分不收敛。
        """

        for sep in _SEPARATORS:
            if sep not in text:
                continue
            head, _, rest = text.partition(sep)
            keep = self._overlap if (self._overlap and len(head) > self._overlap) else 0
            return head, head[len(head) - keep :] + sep + rest
        # 无任何分隔符：按硬长度切，尾部保留 overlap 个字符作为重叠
        return text[: self._size], text[self._size - self._overlap :]


__all__ = ["RecursiveCharChunker"]
__author__ = "晨星"
