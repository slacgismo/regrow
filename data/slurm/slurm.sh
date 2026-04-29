#!/bin/bash
#SBATCH --job-name=herbie_forecasts
#SBATCH --account=pvfleets24
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/kfs2/projects/pvfleets24/repos/regrow/data/slurm_outputs/herbie_%j.out
#SBATCH --error=/kfs2/projects/pvfleets24/repos/regrow/data/slurm_outputs/herbie_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kirsten.perry@nlr.gov

ml mamba
mamba activate /kfs2/projects/pvfleets24/envs/regrow


# Run the script
cd /kfs2/projects/pvfleets24/repos/regrow/data
python generate_herbie_forecasts.py
