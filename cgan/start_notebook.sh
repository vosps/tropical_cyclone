#!/bin/bash

# This script submits a Slurm job to get resources and
# start a Jupyter notebook on those resources.


With Jupyter Notebook cluster, you can run notebook on the local machine and connect to the notebook on the cluster by setting the appropriate port number. Example code:

Go to Server using ssh username@ip_address to server.

Set up the port number for running notebook. On remote terminal run  jupyter notebook --no-browser --port=7800

On your local terminal run ssh -N -f -L localhost:8001:localhost:7800 username@ip_address of server.

Open web browser on local machine and go to http://localhost:8001/