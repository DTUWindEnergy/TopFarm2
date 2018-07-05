@ECHO OFF
ECHO Installing TOPFARM...
REM Create an environment for TOPFARM
conda create -n pyTopfarm python=3.6 --yes
REM Activate the environment
call activate pyTopfarm
REM Install WindIO and DTU's fork of FUSED-Wake
pip install git+https://github.com/FUSED-Wind/windIO.git
pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/FUSED-Wake.git
REM Install TOPFARM
pip install .
REM Switch back to the default environment
call activate base