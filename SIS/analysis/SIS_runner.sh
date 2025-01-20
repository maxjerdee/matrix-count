<<<<<<< HEAD
#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=SIS_runner
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem-per-cpu=5000m 
#SBATCH --time=12:00:00
#SBATCH --account=mejn0
#SBATCH --partition=standard

/bin/hostname
=======
#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=SIS_runner
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem-per-cpu=5000m 
#SBATCH --time=10:00:00
#SBATCH --account=ebruch0
#SBATCH --partition=standard

/bin/hostname
>>>>>>> f25b5b1caa0b2f45b3e011f11766d9572cbc9330
python SIS_runner.py