# GPU 실험 한 번에 돌리기 (윈도우 전용)
#
#   .\scripts\gpu_run.ps1 s04_train_gnn.py --case case118
#
# 하는 일은 네 가지다.
#   1) 최신 코드 받기       git pull
#   2) 실험 실행            .venv\Scripts\python.exe scripts\<스크립트>
#   3) 결과만 커밋          results\ 폴더 (코드는 건드리지 않는다)
#   4) 되돌려 보내기        git push
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
    exit 1
}

$script = $args[0]
$rest = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }
$target = Join-Path $root "scripts\$script"
if (-not (Test-Path $target)) {
    Write-Host "[중단] 그런 스크립트가 없습니다: scripts\$script" -ForegroundColor Red
    exit 1
}

# --- 1) 최신 코드 ---------------------------------------------------------
Write-Host "`n[1/4] 최신 코드 받는 중..." -ForegroundColor Cyan
git pull --ff-only
if ($LASTEXITCODE -ne 0) {
    Write-Host "[중단] git pull 실패. 로컬에 안 올린 변경이 있을 수 있습니다." -ForegroundColor Red
    Write-Host "       git status 로 확인하세요."
    exit 1
}

# --- 2) 실험 -------------------------------------------------------------
Write-Host "`n[2/4] 실험 실행: $script $rest" -ForegroundColor Cyan
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

# --- 3) 결과만 커밋 ------------------------------------------------------
Write-Host "`n[3/4] 결과 커밋 중..." -ForegroundColor Cyan
git add results
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "      새로 생긴 결과 파일이 없습니다. 푸시를 건너뜁니다." -ForegroundColor Yellow
    exit 0
}
$staged | ForEach-Object { Write-Host "      + $_" }
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
git commit -q -m "GPU 실험 결과: $script $rest ($stamp, $mins 분)"

# --- 4) 푸시 -------------------------------------------------------------
Write-Host "`n[4/4] 푸시 중..." -ForegroundColor Cyan
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

Write-Host "`n끝났습니다. 클로드에게 '결과 올렸어' 라고 알려 주세요.`n" -ForegroundColor Green
