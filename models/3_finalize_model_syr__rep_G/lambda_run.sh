#!/bin/bash
echo File \(.py\) to run within tf2py3:
read pyScript
singularity exec --nv ../../../tensorflow/tensorflow-21.07-tf2-py3.sif python $pyScript > lambda_log.txt
