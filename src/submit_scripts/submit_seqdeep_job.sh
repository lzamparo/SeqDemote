#!/bin/bash
#  Batch script for SeqDeep job on the cbio cluster
#  utilizing 1 GPUs, with one thread
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=24:00:00
#
# join stdout and stderr
#PBS -j oe
#
# spool output immediately
#PBS -k oe
#
# specify GPU queue
#PBS -q gpu
#
#  nodes: number of nodes
#  ppn: number of processes per node
#  gpus: number of gpus per node
#  GPUs are in 'exclusive' mode by default, but 'shared' keyword sets them to shared mode.
#  docker: indicator that I want to execute on a node that can run docker. (optional for other ppl)
#  gtxtitan: indicator that I want to execute on nodes that have this particular type of GPU (optional for other ppl)
#PBS -l nodes=1:ppn=1:gpus=1
#
# export all my environment variables to the job
#PBS -V
#
# job name (default = name of script file)
#PBS -N seqdeep_full_training
#
# mail settings (one or more characters)
# email is sent to local user, unless another email address is specified with PBS -M option 
# n: do not send mail
# a: send mail if job is aborted
# b: send mail when job begins execution
# e: send mail when job terminates
#PBS -m n
#
# filename for standard output (default = <job_name>.o<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
##PBS -o myoutput

# Change to working directory used for job submission
cd $PBS_O_WORKDIR

# grab the device number of the GPU assigned to my job
my_gpu=`cat $PBS_GPUFILE`
my_device_num=`echo $my_gpu | cut -c ${#my_gpu}`
my_device="gpu"$my_device_num

# Set THEANO_FLAGS string
echo "Got assigned GPU " $my_device 
export THEANO_FLAGS="device=$my_device" 

cd ~/projects/SeqDemote/src
python train_convnet.py basset_onehot.py
