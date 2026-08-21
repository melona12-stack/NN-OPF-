r"""개발 환경 점검 — 새 컴퓨터에서 제일 먼저 돌린다.

    python scripts/s00_check_env.py

GPU 가 "쓸 수 있다" 는 것과 "실제로 계산이 된다" 는 것은 다르다.

RTX 5060 은 Blackwell 세대라 계산 능력이 ``sm_120`` 인데, PyTorch 를 cu126
이하 빌드로 깔면 그 커널이 아예 컴파일되어 있지 않다. 그런데도
``torch.cuda.is_available()`` 은 **True 를 돌려준다.** 드라이버와 런타임은
멀쩡하니까. 문제는 실제로 행렬 하나를 곱해 볼 때 터진다 —

    RuntimeError: CUDA error: no kernel image is available for execution
    on the device

그래서 이 스크립트는 available 만 보지 않고 **실제 연산을 한 번 시켜 본다.**
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OK, BAD, WARN = "  [정상]", "  [문제]", "  [주의]"


def line(tag: str, msg: str) -> None:
    print(f"{tag} {msg}")


def check_python() -> bool:
    v = sys.version_info
    line(
        OK if (3, 10) <= (v.major, v.minor) < (3, 13) else WARN,
        f"Python {v.major}.{v.minor}.{v.micro} ({platform.system()})",
    )
    if not ((3, 10) <= (v.major, v.minor) < (3, 13)):
        print("        3.11 또는 3.12 를 권합니다 (pandapower 호환).")
    return True


def check_stack() -> bool:
    ok = True
    for name in ("numpy", "scipy", "pandapower", "pytest"):
        try:
            mod = __import__(name)
            line(OK, f"{name} {getattr(mod, '__version__', '?')}")
        except ImportError:
            line(BAD, f"{name} 없음 — pip install -r requirements.txt")
            ok = False
    return ok


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        line(BAD, "torch 없음 — pytorch.org 에서 cu128 이상 빌드를 설치하세요")
        return False

    line(OK, f"PyTorch {torch.__version__}")

    if not torch.cuda.is_available():
        line(WARN, "CUDA 사용 불가 — CPU 로만 돕니다")
        print("        CPU 로도 전부 동작합니다. 다만 학습이 수십 배 느립니다.")
        return True

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    line(OK, f"GPU {name} · sm_{cap[0]}{cap[1]} · VRAM {total / 2**30:.1f} GB")

    # 여기가 이 스크립트의 존재 이유다 — is_available() 은 거짓 안심을 준다.
    try:
        a = torch.randn(512, 512, device="cuda")
        (a @ a).sum().item()
        torch.cuda.synchronize()
        line(OK, "GPU 연산 실제 확인 (행렬곱 512x512)")
    except RuntimeError as e:
        line(BAD, f"GPU 연산 실패: {e}")
        print(
            f"        이 GPU 는 sm_{cap[0]}{cap[1]} 인데 설치된 PyTorch 에 해당 커널이\n"
            "        없습니다. cu128 이상 빌드로 다시 설치하세요:\n"
            "          pip uninstall torch\n"
            "          pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )
        return False

    return True


def check_project() -> bool:
    try:
        from nnopf import load_case, solve_power_flow

        sysm = load_case("case30")
        pf = solve_power_flow(sysm)
        if not pf.converged:
            line(BAD, "case30 조류계산이 수렴하지 않았습니다")
            return False
        line(OK, f"case30 조류계산 수렴 ({pf.iterations}회 반복)")
        return True
    except Exception as e:  # noqa: BLE001
        line(BAD, f"프로젝트 임포트 실패: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("=" * 62)
    print(" NN-OPF 환경 점검")
    print("=" * 62)
    results = [check_python(), check_stack(), check_torch(), check_project()]
    print("-" * 62)
    if all(results):
        print(" 전부 정상입니다. 다음: python -m pytest tests/ -q")
        return 0
    print(" 위 [문제] 항목을 먼저 해결하세요.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
