"""변환기 출력 -> Notion 이 실제로 저장하는 형태.

노션은 마크다운을 블록으로 파싱한 뒤 다시 직렬화하면서 이렇게 정규화한다.

  * 빈 줄을 전부 지운다 (블록 경계가 이미 구조로 표현되므로)
  * 표 안쪽 ``<tr>``/``<td>``/``</tr>`` 의 들여쓰기를 없앤다
  * 콜아웃 안의 표는 ``fit-page-width`` 를 떼고 ``<table header-row="true">`` 로
    (여는/닫는 태그는 탭을 유지, 안쪽 줄만 열 0 으로)
  * 언어를 안 적은 코드블록은 ``javascript`` 로 찍는다
  * 줄마다 따로 ``**`` 를 짝지어 남는 건 버린다

이걸 알아야 update_content 의 old_str 을 정확히 만들 수 있다.
"""
import re


def drop_odd_bold(line: str) -> str:
    """줄 안에서 짝이 안 맞는 ``**`` 를 지운다."""
    pos = [m.start() for m in re.finditer(r"\*\*", line)]
    drop = set(pos[len(pos) - len(pos) % 2:])
    out, i = [], 0
    while i < len(line):
        if i in drop and line[i:i + 2] == "**":
            i += 2
            continue
        out.append(line[i])
        i += 1
    return "".join(out)


_SPAN = re.compile(r"`[^`]*`|\$`[^`]*`\$|\[[^\]]*\]\([^)]*\)")


def escape_specials(line: str) -> str:
    """노션은 본문의 ``| ~ [ ]`` 를 이스케이프해서 돌려준다.

    코드 스팬(``\`x\```), 인라인 수식(``$\`x\`$``), 링크(``[t](u)``) 안은 건드리지
    않는다. 그 안의 문자는 마크다운 특수문자로 해석되지 않기 때문이다.
    """
    out, last = [], 0
    for m in _SPAN.finditer(line):
        out.append(re.sub(r"(?<!\\)([|~\[\]])", r"\\\1", line[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(re.sub(r"(?<!\\)([|~\[\]])", r"\\\1", line[last:]))
    return "".join(out)


def notionize(text: str) -> str:
    out, in_callout, fence = [], False, False
    for line in text.splitlines():
        s = line.lstrip("\t")
        if s.startswith("```"):
            if not fence and s.rstrip() == "```":
                out.append(line + "javascript")
            else:
                out.append(line)
            fence = not fence
            continue
        if fence:
            out.append(line)
            continue
        if line.startswith("<callout"):
            in_callout = True
            out.append(line)
            continue
        if line == "</callout>":
            in_callout = False
            out.append(line)
            continue
        if not line.strip():
            continue                                   # 빈 줄은 사라진다
        if s.startswith("<table"):
            out.append(("\t" if in_callout else "") +
                       ('<table header-row="true">' if in_callout else s))
            continue
        if s.startswith("</table>"):
            out.append(("\t" if in_callout else "") + s)
            continue
        if s.startswith("<td>") and s.endswith("</td>"):
            # 셀 안 텍스트도 본문과 똑같이 이스케이프된다
            out.append("<td>" + escape_specials(drop_odd_bold(s[4:-5])) + "</td>")
            continue
        if s.startswith(("<tr>", "<td>", "</tr>", "<colgroup>", "<col", "</colgroup>")):
            out.append(s)
            continue
        out.append(escape_specials(drop_odd_bold(line)))
    return "\n".join(out)


def main() -> int:
    """``notionize.py <파일.notion.md> [노션에서_받은_본문.txt]``

    인자가 하나면 정규화 결과를 찍고, 둘이면 노션 본문과 줄 단위로 비교해
    **어느 절이 노션에 아직 안 올라갔는지** 보여 준다. Notion MCP 의
    ``update_content`` 는 old_str 이 저장된 형태와 정확히 같아야 하므로,
    이 정규화를 거친 문자열을 써야 한다.
    """
    import difflib
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("사용법: notionize.py <파일.notion.md> [노션본문.txt]", file=sys.stderr)
        return 2

    local = notionize(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if len(sys.argv) == 2:
        print(local)
        return 0

    remote = Path(sys.argv[2]).read_text(encoding="utf-8")
    a, b = remote.splitlines(), local.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    print(f"노션 {len(a)}줄 · 로컬 {len(b)}줄 · 일치율 {sm.ratio():.3f}")
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        print(f"\n[{tag}] 노션[{i1}:{i2}] 로컬[{j1}:{j2}]")
        for ln in a[i1:i1 + 3]:
            print("  노션|", ln[:100])
        for ln in b[j1:j1 + 3]:
            print("  로컬|", ln[:100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
