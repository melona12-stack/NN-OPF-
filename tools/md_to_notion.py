#!/usr/bin/env python
"""GitHub 마크다운 -> Notion-flavored Markdown 변환기.

Notion 은 표준 마크다운과 몇 군데가 다르다.

* 표: ``| a | b |`` 파이프 표를 지원하지 않고 ``<table>`` XML 을 쓴다.
* 인라인 수식: ``$x$`` 가 아니라 백틱을 낀 ``$`x`$`` 형식이다.
* 여러 줄 인용: ``>`` 를 줄마다 쓰면 별개 블록이 되므로 ``<br>`` 로 이어야 한다.
* 페이지 제목: 본문 맨 위의 ``# 제목`` 은 넣지 않는다 (properties 로 전달).
* 강조 상자: GitHub 경고문법 ``> [!NOTE]`` 을 Notion ``<callout>`` 으로 바꾼다.
  이러면 한 소스로 GitHub 와 Notion 양쪽에서 강조 상자가 나온다.

이 스크립트는 ``docs/*.md`` 를 그 규칙에 맞게 바꾼다. 코드블록 안은 건드리지 않는다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# GitHub 경고문법 -> Notion 콜아웃 (아이콘, 배경색)
ALERTS = {
    "NOTE": ("💡", "blue_bg"),
    "TIP": ("✅", "green_bg"),
    "IMPORTANT": ("⭐", "purple_bg"),
    "WARNING": ("⚠️", "yellow_bg"),
    "CAUTION": ("🚨", "red_bg"),
}
_ALERT_RE = re.compile(r"^\[!(" + "|".join(ALERTS) + r")\]\s*(.*)$")


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


def _quote_prefix(line: str) -> tuple[str, str]:
    """``> `` 인용 접두사를 떼어 ``(접두사, 나머지)`` 로 나눈다."""
    if line.startswith("> "):
        return "> ", line[2:]
    if line.rstrip() == ">":
        return "> ", ""
    return "", line


def _convert_tables(lines: list[str]) -> list[str]:
    """파이프 표를 Notion ``<table>`` XML 로 바꾼다.

    **콜아웃(``> [!NOTE]``) 안의 표도 바꾼다.** 이 함수는 ``_merge_quotes``
    보다 먼저 도는데, 그 시점에는 콜아웃 본문이 아직 ``> `` 접두사를 달고
    있다. 접두사를 못 보고 지나치면 파이프 표가 그대로 남고, Notion 은
    파이프 표를 렌더링하지 않아 ``| a | b |`` 가 날것으로 보인다.
    그래서 접두사를 떼고 판정한 뒤 결과에 다시 붙인다.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        pre, body = _quote_prefix(lines[i])
        is_row = body.strip().startswith("|") and body.strip().endswith("|")
        nxt_pre, nxt_body = (
            _quote_prefix(lines[i + 1]) if i + 1 < len(lines) else ("", "")
        )
        if is_row and nxt_pre == pre and _is_separator(nxt_body):
            header = _split_table_row(body)
            i += 2
            rows = []
            while i < len(lines):
                p2, b2 = _quote_prefix(lines[i])
                t = b2.strip()
                if p2 != pre or not (t.startswith("|") and t.endswith("|")):
                    break
                rows.append(_split_table_row(b2))
                i += 1
            out.append(pre + '<table fit-page-width="true" header-row="true">')
            for cells in [header] + rows:
                padded = cells + [""] * (len(header) - len(cells))
                out.append(pre + "\t<tr>")
                for c in padded[: len(header)]:
                    out.append(pre + f"\t\t<td>{c}</td>")
                out.append(pre + "\t</tr>")
            out.append(pre + "</table>")
            continue
        out.append(lines[i])
        i += 1
    return out


def _merge_quotes(lines: list[str]) -> list[str]:
    """인용 블록을 Notion 형식으로 바꾼다.

    ``> [!NOTE]`` 로 시작하면 콜아웃으로, 아니면 ``<br>`` 로 이은 인용문으로.
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        alert = _ALERT_RE.match(buf[0])
        if alert:
            icon, color = ALERTS[alert.group(1)]
            # 제목은 통째로 굵게가 되므로 안쪽 ``**`` 를 지운다. 안 지우면
            # ``**A **B** C**`` 가 되어 Notion 이 별표를 엉뚱하게 짝짓는다.
            head = alert.group(2).strip().replace("**", "")
            body = ([f"**{head}**"] if head else []) + buf[1:]
            out.append(f'<callout icon="{icon}" color="{color}">')
            out.extend("\t" + ln for ln in _join_open_bold(body))
            out.append("</callout>")
        else:
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


_LIST_RE = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)")


def _join_open_bold(body: list[str], struct: tuple[str, ...] = ("<", "```", "|", "$$")) -> list[str]:
    """``**굵게**`` 가 줄바꿈을 넘는 줄을 다음 줄과 이어 붙인다.

    Notion 은 ``**`` 가 줄을 넘어가면 굵게로 인식하지 못하고 별표를 엉뚱한
    자리에 그대로 남긴다. 예를 들어

        **끼워 넣는 자리는 완전히
        동일합니다.** 그래서 ...

    가 ``동일합니다.** 그래서 ... **계약서`` 처럼 깨진다. 원본 마크다운은
    읽기 좋게 80자에서 줄을 접으므로 이런 경우가 계속 생긴다.

    콜아웃 안에서만 생기는 문제가 아니다 — **본문 문단에서도 똑같이 깨진다.**
    그래서 ``convert()`` 가 문서 전체에 한 번, ``_merge_quotes`` 가 콜아웃
    본문에 한 번 돌린다.

    Notion 은 어차피 알아서 줄바꿈하므로, **별표가 안 닫힌 줄만** 다음 줄과
    합쳐 준다. 표·코드블록 같은 구조 줄은 건드리지 않고, 다음 줄이 새 목록
    항목이면 서로 다른 항목을 붙여 버리므로 거기서 멈춘다.
    """
    out: list[str] = []
    i = 0
    while i < len(body):
        line = body[i]
        # 별표 개수가 홀수면 굵게가 이 줄에서 안 닫혔다는 뜻
        while (line.count("**") % 2 == 1 and i + 1 < len(body)
               and body[i + 1].strip()
               and not body[i + 1].lstrip().startswith(struct)
               and not _LIST_RE.match(body[i + 1])
               and not line.lstrip().startswith(struct)):
            i += 1
            line = line.rstrip() + " " + body[i].lstrip()
        out.append(line)
        i += 1
    return out


def _convert_inline_math(text: str) -> str:
    """``$x$`` -> ``$`x`$``. 이미 백틱이 있으면 건드리지 않는다."""
    def repl(m: re.Match[str]) -> str:
        body = m.group(1)
        if body.startswith("`") and body.endswith("`"):
            return m.group(0)
        return f"$`{body}`$"

    return _INLINE_MATH_RE.sub(repl, text)


_DOC_LINK_RE = re.compile(r"\]\((?:\.\./)?(?:docs/)?(\d\d_[a-z_]+\.md)(#[^)]*)?\)")


def _rewrite_doc_links(text: str, links: dict[str, str]) -> str:
    """문서 간 상대링크(``](01_x.md)``)를 Notion 페이지 URL 로 바꾼다.

    Notion 에는 ``docs/`` 디렉터리가 없으므로 상대링크가 그대로면 깨진다.
    매핑에 없는 파일은 손대지 않는다.
    """
    def repl(m: re.Match[str]) -> str:
        url = links.get(m.group(1))
        return f"]({url})" if url else m.group(0)

    return _DOC_LINK_RE.sub(repl, text)


def convert(md: str, links: dict[str, str] | None = None) -> tuple[str, str]:
    """마크다운 문서를 ``(제목, Notion 본문)`` 으로 변환한다."""
    if links:
        md = _rewrite_doc_links(md, links)
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

    # 본문 문단의 줄 넘는 굵게를 먼저 잇는다. 콜아웃(``>``)·표(``|``) 줄은
    # 각자 뒤에서 처리하므로 여기서는 건드리지 않는다.
    staged = _join_open_bold(staged, ("<", "```", "|", "$$", ">", "#", "\x00"))
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

    link_file = Path(__file__).resolve().parent.parent / "docs" / "notion_links.json"
    links: dict[str, str] = {}
    if link_file.exists():
        raw = json.loads(link_file.read_text(encoding="utf-8"))
        links = {k: v for k, v in raw.items() if not k.startswith("_")}

    for path in sys.argv[1:]:
        if path.endswith(".notion.md"):
            continue  # 이미 변환된 산출물 (glob 로 딸려 들어온 경우)
        title, body = convert(Path(path).read_text(encoding="utf-8"), links)
        out = Path(path).with_suffix(".notion.md")
        out.write_text(body, encoding="utf-8")
        print(f"{path} -> {out}  (제목: {title})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
