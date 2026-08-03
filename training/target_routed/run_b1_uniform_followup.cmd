@echo off
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=D:\2026\test\venv310\Scripts\python.exe"
set "STAGE=%~1"
set "GPU_ID=%~2"
set "EXTRA_ARGS="

if "%STAGE%"=="" set "STAGE=all"
if "%GPU_ID%"=="" set "GPU_ID=0"
if /I "%~3"=="force" set "EXTRA_ARGS=--force"
set "CUDA_VISIBLE_DEVICES=%GPU_ID%"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 2
)

echo [B1 FOLLOW-UP] stage=%STAGE% GPU=%GPU_ID%
echo [REUSE] Existing completed B0 and B2 checkpoints are not retrained.
echo [TRAIN] B1 uniform direct concat only, seeds 42/43/44.
echo [COMPARE] B0 Context-only vs B1 all-target PPG vs B2 decoupled final.
echo [IMPORTANT] Run only after run_target_routed_final.cmd has completed.

"%PYTHON_EXE%" "%SCRIPT_DIR%run_b1_uniform_followup.py" --stage "%STAGE%" %EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo [FAILED] B1 follow-up exited with code %EXIT_CODE%.
  exit /b %EXIT_CODE%
)

if /I "%STAGE%"=="smoke" (
  echo [DONE] %SCRIPT_DIR%reports\b0_b1_b2\B1_SMOKE_AUDIT.json
) else (
  echo [DONE] %SCRIPT_DIR%reports\b0_b1_b2\B0_B1_B2_RESULTS.md
)
exit /b 0
