# GPU 실험 한 번에 돌리기 (윈도우 전용)
#
#   .\scripts\gpu_run.ps1 s03_train_surrogate.py --case case118
#
# 하는 일은 네 가지다.
#   1) 최신 코드 받기       git pull
#   2) 실험 실행            .venv\Scripts\python.exe scripts\<스크립트>
#   3) 결과만 커밋          results\ 폴더 (코드는 건드리지 않는다)
#   4) 되돌려 보내기        git push
#
# 작업 목록(.vscode/tasks.json)에서 실험을 직접 돌렸다면 2번은 이미 끝났다.
# 그때는 커밋·푸시만 하면 된다.
#
#   .\scripts\gpu_run.ps1 --commit-only
#
# 결과를 복사해 붙여넣는 대신 저장소로 돌려보내는 이유는 단순하다 —
# 로그를 긁다 보면 숫자가 잘리거나 섞이는데, JSON 을 그대로 받으면 그럴 일이 없다.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[중단] 가상환경이 없습니다: $py" -ForegroundColor Red
    Write-Host "       py -m venv .venv  부터 하세요."
    exit 1
}

if ($args.Count -eq 0) {
    Write-Host "[사용법] .\scripts\gpu_run.ps1 <스크립트이름> [옵션...]" -ForegroundColor Yellow
    Write-Host "  예)    .\scripts\gpu_run.ps1 s03_train_surrogate.py --case case118"
    Write-Host "  또는)  .\scripts\gpu_run.ps1 --commit-only   (이미 돌린 결과만 올리기)"
    exit 1
}

# 이미 돌린 결과만 올리는 모드. 작업 목록에서 실험을 직접 돌렸을 때 쓴다.
$commitOnly = ($args[0] -eq "--commit-only")

if (-not $commitOnly) {
    $script = $args[0]
    $rest = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }
    $target = Join-Path $root "scripts\$script"
    if (-not (Test-Path $target)) {
        Write-Host "[중단] 그런 스크립트가 없습니다: scripts\$script" -ForegroundColor Red
        exit 1
    }
}

$mins = 0
$nStep = if ($commitOnly) { 3 } else { 4 }

if ($commitOnly) {
    Write-Host "`n[1/$nStep] 이미 돌린 결과만 올립니다 (git pull · 실험 건너뜀)." -ForegroundColor Cyan
} else {
    # --- 1) 최신 코드 -----------------------------------------------------
    Write-Host "`n[1/$nStep] 최신 코드 받는 중..." -ForegroundColor Cyan
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[중단] git pull 실패. 로컬에 안 올린 변경이 있을 수 있습니다." -ForegroundColor Red
        Write-Host "       git status 로 확인하세요."
        exit 1
    }

    # --- 2) 실험 ----------------------------------------------------------
    Write-Host "`n[2/$nStep] 실험 실행: $script $rest" -ForegroundColor Cyan
    Write-Host "      (오래 걸립니다. 창을 닫지 마세요.)`n"
    $started = Get-Date
    & $py $target @rest
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

    if ($code -ne 0) {
        Write-Host "`n[중단] 실험이 실패했습니다 (종료코드 $code, $mins 분 경과)." -ForegroundColor Red
        Write-Host "       위 에러 메시지를 그대로 알려 주세요."
        exit $code
    }
    Write-Host "`n      완료 ($mins 분)" -ForegroundColor Green
}

# --- 3) 결과만 커밋 ------------------------------------------------------
Write-Host "`n[$($nStep - 1)/$nStep] 결과 커밋 중..." -ForegroundColor Cyan

# 커밋하려면 git 이 "누가 만든 것인지" 를 알아야 한다. 이게 없으면 commit 이
# 조용히 실패하고, 푸시는 "Everything up-to-date" 를 찍고 끝난다 - 성공한 것처럼
# 보이지만 아무것도 안 올라간다. 먼저 확인한다.
$who = (git config user.email)
if (-not $who) {
    Write-Host "[중단] 이 컴퓨터에 git 사용자 정보가 없어 커밋할 수 없습니다." -ForegroundColor Red
    Write-Host "       아래 두 줄을 터미널에 한 번만 붙여 넣으세요 (자기 것으로 바꿔서):"
    Write-Host '         git config --global user.name  "이름"' -ForegroundColor Yellow
    Write-Host '         git config --global user.email "메일주소"' -ForegroundColor Yellow
    Write-Host "       그 다음 이 작업을 다시 돌리면 됩니다."
    exit 1
}

git add results
if ($LASTEXITCODE -ne 0) {
    Write-Host "[중단] git add 실패." -ForegroundColor Red
    exit 1
}

$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "      커밋할 결과 파일이 없습니다." -ForegroundColor Yellow
    Write-Host "      results\ 폴더의 지금 상태는 이렇습니다:"
    git status --short results
    Write-Host "      (아무것도 안 나오면 이미 다 올라가 있다는 뜻입니다.)"
    exit 0
}
$staged | ForEach-Object { Write-Host "      + $_" }

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$what = if ($commitOnly) { "작업 목록에서 직접 실행" } else { "$script $rest ($mins 분)" }
git commit -q -m "GPU 실험 결과: $what ($stamp)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[중단] git commit 실패. 위 메시지를 그대로 알려 주세요." -ForegroundColor Red
    exit 1
}

# --- 4) 푸시 -------------------------------------------------------------
Write-Host "`n[$nStep/$nStep] 푸시 중..." -ForegroundColor Cyan
$branch = git rev-parse --abbrev-ref HEAD
for ($i = 1; $i -le 4; $i++) {
    git push -u origin $branch
    if ($LASTEXITCODE -eq 0) { break }
    if ($i -eq 4) {
        Write-Host "[중단] 푸시 4회 실패. 네트워크를 확인하세요." -ForegroundColor Red
        exit 1
    }
    $wait = [math]::Pow(2, $i)
    Write-Host "      실패. $wait 초 후 재시도..." -ForegroundColor Yellow
    Start-Sleep -Seconds $wait
}

$head = git rev-parse --short HEAD
Write-Host "`n끝났습니다. 올라간 커밋: $head" -ForegroundColor Green
Write-Host "클로드에게 '결과 올렸어' 라고 알려 주세요.`n" -ForegroundColor Green
