# Building and deploying the genslm container
For the deployment of genslm we used apptainer. Minimal steps on how to build and utilize the container are below.

clone the genslm repo:

```
git clone https://github.com/ramanathanlab/genslm.git
```
Replace the existing `requirements.txt` at:

`genslm/requirments/requirments.txt`

With the copy from this repo.

```
cp requirements.txt genslm/requirments/requirments.txt
```
Replace the existing `setup.cfg` at:

`genslm/setup.cfg`

With the copy from this repository

```
cp setup.cfg genslm/setup.cfg
```
# Building the container

Download the `genslm.def` file from this repository to a system where you have root access.

To build the container, run:

```
sudo apptainer build genslm.sif genslm.def

```
# Running the Container

Once the container has been built, you can run the GenSLM software via:

```
apptainer exec --nv --bind /scratch /optnfs/singularity/genslm.sif python my.python.cod.sh
```
