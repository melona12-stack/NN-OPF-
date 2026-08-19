#!/usr/bin/env python
"""GitHub 마크다운 -> Notion-flavored Markdown 변환기.

Notion 은 표준 마크다운과 몇 군데가 다르다.

* 표: ``| a | b |`` 파이프 표를 지원하지 않고 ``<table>`` XML 을 쓴다.
* 인라인 수식: ``$x$`` 가 아니라 백틱을 낀 ``$`x`$`` 형식이다.
* 여러 줄 인용: ``>`` 를 줄마다 쓰면 별개 블록이 되므로 ``<br>`` 로 이어야 한다.
* 페이지 제목: 본문 맨 위의 ``# 제목`` 은 넣지 않는다 (properties 로 전달).

이 스크립트는 ``docs/*.md`` 를 그 규칙에 맞게 바꾼다. 코드블록 안은 건드리지 않는다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _split_table_row(line: str) -> list[str]:
    """``| a | b |`` 한 줄을 셀 목록으로 나눈다 (이스케이프된 ``\\|`` 는 유지)."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells, cur, esc = [], "", False
    for ch in body:
        if esc:
            cur += "|" if ch == "|" else "\\" + ch
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "|":
            cells.append(cur.strip())
            cur = ""
        else:
            cur += ch
    cells.append(cur.strip())
    return cells


_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


def _is_separator(line: str) -> bool:
    """``|---|---|`` 형태의 구분선인지."""
    return bool(_SEP_RE.match(line)) and "-" in line


def _convert_tables(lines: list[str]) -> list[str]:
    """파이프 표를 Notion ``<table>`` XML 로 바꾼다."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_row = line.strip().startswith("|") and line.strip().endswith("|")
        if is_row and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            header = _split_table_row(line)
            i += 2
            rows = []
            while i < len(lines):
                nxt = lines[i].strip()
                if not (nxt.startswith("|") and nxt.endswith("|")):
                    break
                rows.append(_split_table_row(lines[i]))
                i += 1
            out.append('<table fit-page-width="true" header-row="true">')
            for cells in [header] + rows:
                padded = cells + [""] * (len(header) - len(cells))
                out.append("\t<tr>")
                for c in padded[: len(header)]:
                    out.append(f"\t\t<td>{c}</td>")
                out.append("\t</tr>")
            out.append("</table>")
            continue
        out.append(line)
        i += 1
    return out


def _merge_quotes(lines: list[str]) -> list[str]:
    """연속된 ``>`` 줄을 ``<br>`` 로 이어 하나의 인용 블록으로 만든다."""
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append("> " + "<br>".join(buf))
            buf.clear()

    for line in lines:
        if line.startswith(">"):
            text = line[1:].strip()
            if text:
                buf.append(text)
            else:
                buf.append("")
        else:
            flush()
            out.append(line)
    flush()
    return [ln for ln in out if ln.strip() != ">"]


_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)((?:\\.|[^$\\])+?)(?<!\\)\$(?!\$)")


def _convert_inline_math(text: str) -> str:
    """``$x$`` -> ``$`x`$``. 이미 백틱이 있으면 건드리지 않는다."""
    def repl(m: re.Match[str]) -> str:
        body = m.group(1)
        if body.startswith("`") and body.endswith("`"):
            return m.group(0)
        return f"$`{body}`$"

    return _INLINE_MATH_RE.sub(repl, text)


def convert(md: str) -> tuple[str, str]:
    """마크다운 문서를 ``(제목, Notion 본문)`` 으로 변환한다."""
    lines = md.splitlines()

    # 맨 위 H1 을 제목으로 떼어낸다.
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]

    # 코드블록/수식블록 안은 변환 대상에서 제외한다.
    protected: list[str] = []

    def protect(block: list[str]) -> str:
        protected.append("\n".join(block))
        return f"\x00PROTECTED{len(protected) - 1}\x00"

    staged: list[str] = []
    buf: list[str] = []
    fence: str | None = None
    for line in lines:
        stripped = line.strip()
        if fence is None:
            if stripped.startswith("```"):
                fence = "```"
                buf = [line]
            elif stripped == "$$":
                fence = "$$"
                buf = [line]
            else:
                staged.append(line)
        else:
            buf.append(line)
            closed = stripped.startswith("```") if fence == "```" else stripped == "$$"
            if closed:
                staged.append(protect(buf))
                fence, buf = None, []
    if buf:
        staged.extend(buf)

    staged = _convert_tables(staged)
    staged = _merge_quotes(staged)
    staged = [_convert_inline_math(ln) for ln in staged]

    body = "\n".join(staged)
    for idx, block in enumerate(protected):
        body = body.replace(f"\x00PROTECTED{idx}\x00", block)
    return title, body


def main() -> int:
    if len(sys.argv) < 2:
        print("사용법: md_to_notion.py <파일.md> [...]", file=sys.stderr)
        return 2
    for path in sys.argv[1:]:
        title, body = convert(Path(path).read_text(encoding="utf-8"))
        out = Path(path).with_suffix(".notion.md")
        out.write_text(body, encoding="utf-8")
        print(f"{path} -> {out}  (제목: {title})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
