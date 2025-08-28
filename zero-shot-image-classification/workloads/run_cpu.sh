./run_all_task_cpu.sh --model_dtype bfloat16 --warm_up_steps 10 --run_steps 10 2>&1 | tee cpu_output_OOB.log
./run_all_task_cpu.sh --model_dtype bfloat16 --warm_up_steps 10 --run_steps 10 --torch_compile True --compare_outputs True 2>&1 | tee cpu_output_compile.log

python analyse_logs.py --file_names cpu_output_OOB.log,cpu_output_compile.log --out_name cpu_benchmark.log
