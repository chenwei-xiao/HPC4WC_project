# A Python Implementation of GFS Scale-Aware Mass-Flux Shallow Convection Scheme Module
- `funcphys.py`: thermodynamic functions
- `physcons.py`: Constants
- `samfaerosols.py`: Aerosol process
- `samfshalcnv.py`: shalcnv scheme
- `serialization.py`: Serialization

## Storage order in GT4Py
- 1D array: (1, nx, 1)
- 2D array: (1, nx, nz)

## Build with docker in Linux
execute `build.sh` then `enter.sh`.

## Build with docker in Windows
1. execute `docker build -t hpc4wc_project .`
2. execute `docker run -i -t --rm --mount type=bind,source={ABSOLUTE PATH OF THIS FOLDER},target=/work --name=hpc4wc_project hpc4wc_project`

## Run on Piz Diant
execute `env_diant`

## Tests
execute `pytest tests.py`
