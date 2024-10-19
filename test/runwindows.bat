@echo off
setlocal EnableDelayedExpansion

REM This script is based on the assumption you execute it from the 'test' directory

REM Default exponent range
set n_start=0
set n_end=16

REM Create a timestamp to uniquely identify output files (optional)
REM For simplicity, we will skip timestamp_hash in this script

REM Directory to store output files
set output_dir=.\results\stdout
if not exist "%output_dir%" (
    mkdir "%output_dir%"
)

REM Prefer release but use debug otherwise
set executable_path=..\build\Release\CUDASIMULATEWORLDS.exe
if not exist "%executable_path%" (
    set executable_path=..\build\Debug\CUDASIMULATEWORLDS.exe
)

REM Optional args for logging world, off for benchmark runs
set world_log_idx=-1
set simdata_output_dir=.\results\simdata

REM Loop over each world count
for /L %%n in (%n_start%,1,%n_end%) do (
    set /a "w=1<<%%n"
    echo ---------------------------
    echo Running with !w! Worlds:
    
    set output_file=%output_dir%\out_!w!.txt

    if "!world_log_idx!"=="-1" (
        REM Run the simulation without world logging
        "%executable_path%" !w! >> "!output_file!" 2>&1
    ) else (
        REM Run the simulation with world logging enabled
        "%executable_path%" !w! !world_log_idx! !simdata_output_dir! >> "!output_file!" 2>&1
    )

    echo Finished running with !w! Worlds
)
echo Finished all simulations

endlocal
