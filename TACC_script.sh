#!/bin/bash
#SBATCH -J transfer     # Job name
#SBATCH -o out%j # Name of stdout output file(%j expands to jobId)
#SBATCH -e err%j # Name of stderr output file(%j expands to jobId)
#SBATCH -p v100                  # Submit to the 'normal' or 'development' queue
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 24:00:00             # Max run time (hh:mm:ss) - 72 hours; this is determined by the queue you submit to, in this case gpu-long; please refer to the user guide for queue info.
#SBATCH --mail-user=jiyang.zhang@utexas.edu
#SBATCH --mail-type=ALL
# The next line is required if the user has more than one project
#SBATCH -A compdisc      # Allocation name to charge job against

module reset
source ../anaconda3/etc/profile.d/conda.sh
conda activate com
module load python3
module load gcc
module load intel/17.0.4
module load cuda/10.0 cudnn/7.6.2 nccl/2.4.7
cd translate/
python __main__.py ../config/transfer.yaml --decode ../data/test/test.token.code ../data/test/test.token.api
