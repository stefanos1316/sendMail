#!/bin/sh

# A script that executes all of our expriments
# and collects the required measurements

# Text fonts for Linux distros
bold=$(tput bold)
underline=$(tput smul)
default=$(tput sgr0)
greenlabel=$(tput setab 2)
redlabel=$(tput setab 1)
yellowlabel=$(tput setab 3)

# Set default values
repetitions=1

# Help
help_info()
{
  echo "-r <repeitions number> or --repetitions <repeitions number> are used to define the number of repetitions to run each task"
  exit
}

# Log with a timestamp
log()
{
  # Output is redirected to the log file if needed at the script's lop level
  date +'%F %T ' | tr -d \\n 1>&2
  echo "$@" 1>&2
}

# Function that executes
# $1 is the name of the task (e.g., Transformer-XL)
# $2 is the name of the framework (e.g., PyTorch)
# $3 is the command to execute a task for the corresponding task ($1) that is written with a framework ($2)
perf_command()
{
  log "Obtaining energy and run-time performance measurements"
  perf stat -r "$repetitions" -e power/energy-pkg/,power/energy-ram/ "$2" 2> "$measurements"/."$1"_"$2".txt
}

# Get command-line arguments
OPTIONS=$(getopt -o r: --long repetitions: -n 'run_experiments' -- "$@")
eval set -- "$OPTIONS"
while true; do
  case "$1" in
    -r|--repetitions) repetitions="$2"; shift 2;;
    -h|--help) help_info; shift;;
    --) shift; break;;
    *) >&2 log "${redlabel}[ERROR]${default} Wrong command line argument, please try again."; exit 1;;
  esac
done

# Set measurements directory
if [ ! -d measurements ]; then
  mkdir measurements
  log "Created directory measurements"
  measurements=$(echo "$PWD/measurements")
fi

# Go into the DeepLearning Examples repository
cd DeepLearningExamples

# Declare the array of test cases
declare -a arr=("PyTorch" "TensorFlow")

# Execute Transformer-XL for PyToch and TensorFlow
for i in "${arr[@]}"; do
  log "Executing Transofrmer-XL for $i"
  cd "$i"/LanguageModeling/Transformer-XL
  perf_command "transformer_xl" "$i" "docker run --gpus all --init -it --rm --network=host --ipc=host -v $PWD:/workspace/transformer-xl transformer-xl bash run_wt103_base.sh train 1"
  cd ../../
done

log "Done with all tasks, sending email to the corresponding author"
exit