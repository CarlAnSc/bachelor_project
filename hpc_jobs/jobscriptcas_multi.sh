# LSBATCH: User input
#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J train
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 0:15
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
~/miniconda3/envs/hpc_env/bin/python \
src/models/train_model_multilabel_cas.py --path /work3/s204162/data/TopFactor/ \
--epochs 50 --batch_size 128 --lr 1e-4 --weight_decay 0.0005 \
--momentum 0.9 --optimizer sgd --weighted_loss True