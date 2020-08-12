# A Python Implementation of GFS Scale-Aware Mass-Flux Shallow Convection Scheme Module
- `funcphys.py`: thermodynamic functions
- `physcons.py`: Constants
- `samfaerosols.py`: Aerosol process
- `samfshalcnv.py`: shalcnv scheme
- `serialization.py`: Serialization
- `kernels/stencils_*.py`: GT4Py stencils of the shalcnv scheme
- `kernels/utils.py`: useful functions for GT4Py arrays

## Storage order in GT4Py
- 1D array: (1, nx, 1)
- 2D array: (1, nx, nz)

## Build with docker in Linux
execute `build.sh` then `enter.sh`.

## Build with docker in Windows
1. execute `docker build -t hpc4wc_project .`
2. execute `docker run -i -t --rm --mount type=bind,source={ABSOLUTE PATH OF THIS FOLDER},target=/work --name=hpc4wc_project hpc4wc_project`
3. execute `ipython main.py` or `benchmark.py`

## Run on Piz Diant
1. CHANGE `ISDOCKER` to False and `DATAPATH` in `shalconv/__init__.py`
2. execute `source env_diant`
3. execute `ipython main.py` or `benchmark.py`

## Tests
Inside tests folder, execute `ipython run_serialization.py` to generate serialization
data needed for tests, then execute `ipython test_*.py` to run tests.
