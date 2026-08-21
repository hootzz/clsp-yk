@echo off
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
if not defined PYTHON_EXE set "PYTHON_EXE=python"
set "STAGE=%~1"
set "GPU_ID=%~2"
set "EXTRA_ARGS="

if "%STAGE%"=="" set "STAGE=all"
if "%GPU_ID%"=="" set "GPU_ID=0"
if /I "%~3"=="force" set "EXTRA_ARGS=--force"
set "CUDA_VISIBLE_DEVICES=%GPU_ID%"

where "%PYTHON_EXE%" >nul || (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 2
)

echo [TARGET-ROUTED TITLE FINAL] stage=%STAGE% GPU=%GPU_ID%
echo [GPU ONLY] CPU fallback is disabled.
echo [VALENCE] frozen title-level Context-only + chronological EB offset.
echo [AROUSAL/COG] residual-free direct Context+PPG + chronological EB offset.
echo [CONTROL] matched title-level Context-only, same splits/seeds/training budget.
echo [EXTERNAL] WESAD V/A, CogWear C, EmoWear V/A, and VRFS diagnostics.

"%PYTHON_EXE%" "%SCRIPT_DIR%run_target_routed_final.py" --stage "%STAGE%" %EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [FAILED] target-routed experiment exited with code %EXIT_CODE%.
  exit /b %EXIT_CODE%
)

if /I "%STAGE%"=="smoke" (
  echo [DONE] GPU/architecture smoke test passed.
  echo [AUDIT] %SCRIPT_DIR%reports\EXPERIMENT_CONTRACT.json
) else if /I "%STAGE%"=="train" (
  echo [DONE] Full training completed. Run the evaluate stage for final metrics.
) else if /I "%STAGE%"=="external" (
  echo [DONE] %SCRIPT_DIR%reports\external_zero_shot\TARGET_ROUTED_EXTERNAL_ZERO_SHOT_RESULTS.md
) else (
  echo [DONE] %SCRIPT_DIR%reports\TARGET_ROUTED_FINAL_RESULTS.md
)
exit /b 0
