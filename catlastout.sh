files=$(ls -rt slurm-?????.out)
file=$(echo $files | rev | cut -d ' ' -f 1 | rev)
echo $file && less -r +F $file
