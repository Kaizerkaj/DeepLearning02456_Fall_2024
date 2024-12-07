#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuVal
### -- st the job Name -- 
#BSUB -J MNISTparamE
### -- ask for number of cores (default: 1) -- 
#BSUB -n 32
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=8:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=10GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 11GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u kaizerkajj@gmail.com
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
# myapplication.x input.in > output.out

python NewModels.py param MNIST