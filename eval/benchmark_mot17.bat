@echo off
setlocal

cd /d "%~dp0.."

python -m eval.benchmark_mot17 ^
  --mot-root data\mot17\val ^
  --results-root results\mot17_benchmark ^
  --trackers sort bytetrack ocsort botsort ^
  --detectors DPM FRCNN SDP ^
  --conf-thresh 0.1 ^
  --no-gmc ^
  %*

endlocal
