r"""결과를 그림으로 — 숫자 표만 보고는 "왜" 를 알 수 없다.

우리가 지금까지 낸 결론 중 표로는 안 보이고 그림으로만 보이는 게 여럿 있다.
예를 들어 06 문서 §4.1 의 "열화가 미지 N-1 표본에만 뭉쳐 있다" 는 평균값
5.16% 만 봐서는 절대 안 보인다. 표본별 분포를 그려야 두 봉우리가 갈라진다.

색은 눈대중으로 고르지 않았다. 세 계열 모두 색각이상(적록·청황) 분리도와
배경 대비를 검증기로 통과시킨 값이다. 밝은 배경에서 청록은 대비가 3:1 을
못 넘겨서, 쓸 때는 반드시 직접 라벨을 같이 단다.

한글 폰트가 없는 환경(리눅스 컨테이너 등)에서는 축·제목이 네모로 깨진다.
``setup()`` 이 쓸 수 있는 폰트를 찾아보고, 없으면 영문 라벨로 자동 전환한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경에서도 저장은 된다
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["Theme", "setup", "KO", "save"]


class Theme:
    """검증된 팔레트 한 벌. ``light`` / ``dark`` 두 가지."""

    def __init__(self, dark: bool = False) -> None:
        if dark:
            self.surface = "#1a1a19"
            self.ink = "#ffffff"
            self.ink2 = "#c3c2b7"
            self.grid = "#383835"
            self.series = ["#3987e5", "#d95926", "#199e70"]
            self.critical = "#d03b3b"
        else:
            self.surface = "#fcfcfb"
            self.ink = "#0b0b0b"
            self.ink2 = "#52514e"
            self.grid = "#e5e4e0"
            self.series = ["#2a78d6", "#eb6834", "#1baf7a"]
            self.critical = "#d03b3b"
        self.dark = dark


# 한글 폰트가 없을 때 쓸 영문 대체 라벨
KO = True


def setup(dark: bool = False) -> Theme:
    """matplotlib 전역 스타일을 잡고 테마를 돌려준다."""
    global KO
    have = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"):
        if cand in have:
            # 폰트를 **목록**으로 준다. 한글 폰트 대부분은 진짜 빼기 기호
            # (U+2212) 를 갖고 있지 않은데, 로그 눈금의 지수(10⁻⁵)가 그걸 쓴다.
            # axes.unicode_minus=False 는 일반 눈금만 덮고 수식 눈금은 못 덮는다.
            # 뒤에 DejaVu Sans 를 붙여 두면 없는 글자만 거기서 가져온다.
            plt.rcParams["font.family"] = [cand, "DejaVu Sans"]
            KO = True
            break
    else:
        KO = False  # 한글 폰트 없음 — 호출부가 영문 라벨을 쓰도록 알린다

    t = Theme(dark)
    plt.rcParams.update({
        "figure.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "text.color": t.ink,
        "axes.labelcolor": t.ink2,
        "axes.edgecolor": t.grid,
        "xtick.color": t.ink2,
        "ytick.color": t.ink2,
        "grid.color": t.grid,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,          # 격자는 데이터 뒤로 — 눈에 안 띄어야 한다
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "lines.linewidth": 2.0,          # 얇은 마크
        "legend.frameon": False,
        "font.size": 11,
        "figure.dpi": 130,
    })
    return t


def save(fig, path: str | Path) -> Path:
    """저장하고 경로를 돌려준다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def label(ko: str, en: str) -> str:
    """한글 폰트가 있으면 한글, 없으면 영문."""
    return ko if KO else en


def annotate_best(ax, x: float, y: float, text: str, t: Theme) -> None:
    """한 점만 골라 직접 라벨 — 모든 점에 숫자를 달지 않는다."""
    ax.plot([x], [y], "o", ms=8, color=t.critical, zorder=5)
    ax.annotate(
        text, (x, y), textcoords="offset points", xytext=(8, 8),
        color=t.ink, fontsize=10, fontweight="bold",
    )


def pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")


def thousands(ax) -> None:
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")


def bar_gap(ax) -> None:
    """막대 사이에 배경색 틈을 둔다 — 붙어 있으면 경계가 안 보인다."""
    for patch in ax.patches:
        patch.set_linewidth(1.5)
        patch.set_edgecolor(plt.rcParams["axes.facecolor"])


def annotate_bars(ax, fmt: str = "{:.2f}", t: Theme | None = None) -> None:
    """막대 끝에 값을 직접 적는다. 청록처럼 대비가 낮은 색을 쓸 때 필수."""
    ink = t.ink if t else "#0b0b0b"
    for p in ax.patches:
        h = p.get_height()
        if not np.isfinite(h):
            continue
        ax.annotate(
            fmt.format(h), (p.get_x() + p.get_width() / 2, h),
            ha="center", va="bottom", xytext=(0, 3),
            textcoords="offset points", fontsize=9, color=ink,
        )
