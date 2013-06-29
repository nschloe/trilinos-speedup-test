#!/bin/bash

# Make sure CMake is loaded.
command -v cmake >/dev/null 2>&1 || { echo "cmake not found. Try 'module load cmake'." >&2; exit 1;}

# Check which compiler is loadad.
if [ "$CRAY_PRGENVPGI" == "loaded" ]; then
  COMPILER_NAME="pgi"
elif [ "$CRAY_PRGENVCRAY" == "loaded" ]; then
  COMPILER_NAME="cray"
elif [ "$CRAY_PRGENVGNU" == "loaded" ]; then
  COMPILER_NAME="gnu"
elif [ "$CRAY_PRGENVINTEL" == "loaded" ]; then
  COMPILER_NAME="intel"
elif [ "$CRAY_PRGENVPATHSCALE" == "loaded" ]; then
  COMPILER_NAME="pathscale"
else
  echo "Unknown compiler suite selected. Abort."
  exit 1
fi

# Give the user some time to CTRL-C out.
echo "Using <$COMPILER_NAME> compilers."
sleep 5

CXX=CC \
CMAKE_PREFIX_PATH="${SCRATCH}/trilinos/dev/$COMPILER_NAME/:${CMAKE_PREFIX_PATH}" \
cmake \
  ${HOME}/software/trilinos-speedup-test/dev/
