# LSBATCH: User input
#!/bin/sh
### General options
### -- set the job Name --
#BSUB -J train-JuCa
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 16:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=4GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo hpc_jobs/job_output/gpu_%J.out
#BSUB -eo hpc_jobs/job_output/gpu_%J.err
# -- end of LSF options --

# Load the cuda module
module load cuda/11.7
# Which python
cd src/models/
~/miniconda3/envs/hpc_env/bin/python \
use_embeddings.py --classifier Xgb --tsne yea #logistic-regr #Xgb
