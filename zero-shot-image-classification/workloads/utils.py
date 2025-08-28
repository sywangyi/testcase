import os
import re
import glob
from datetime import datetime

# Add a fixed list of environment variables to filter
env_vars_filter = ["LD_LIBRARY_PATH", "PATH", "LS_COLORS", "HUGGING_FACE_HUB_TOKEN", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]

# Modify the filter_env_vars function to use the fixed list
def filter_env_vars(env_vars, env_vars_filter):
    """
    Filter out environment variables based on a fixed list.
    """
    return {key: value for key, value in env_vars.items() if key not in env_vars_filter}

# Extract environment variables
def extract_environment_variables(content):
    env_vars_pattern = re.compile(r"Environment Variables:\n(.*?)\nBASH_FUNC_run_task", re.DOTALL)
    match = env_vars_pattern.search(content)
    if match:
        env_vars = match.group(1)
        env_vars_dict = {}
        for line in env_vars.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars_dict[key.strip()] = value.strip()
        return filter_env_vars(env_vars_dict, env_vars_filter)
    return {}

# Extract Python package versions
def extract_python_packages(content):
    pattern = re.compile(r"Python Package Versions:\n(.*?)\nExecuting command:", re.DOTALL)
    match = pattern.search(content)
    if match:
        packages_section = match.group(1)
        packages = {}
        for line in packages_section.splitlines():
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    package_name = parts[0]
                    package_version = parts[1]
                    packages[package_name] = package_version
        return packages
    return {}

# Compare package versions across models
def compare_packages(all_packages):
    reference = None
    discrepancies = {}
    for model, packages in all_packages.items():
        if reference is None:
            reference = packages
        else:
            for package, version in packages.items():
                if package in reference and reference[package] != version:
                    discrepancies[package] = True
    return discrepancies

def extract_with_patterns(content, patterns):
    for pattern in patterns:
        match = pattern.search(content)
        if match:
            return match.group(1)
    return None

def parse_inference_directory(log_dir):
    # Initialize data structures
    performance_data = []
    error_logs = {}
    all_packages = {}

    # Regex patterns for extracting data
    total_time_pattern = re.compile(r"total_time \[ms\]: \[([^\]]+)\]")
    pipeline_avg_time_patterns = [
        re.compile(r"pipeline average time \[ms\]:? ([\d.]+)"),  # Old format
        re.compile(r"pipeline_average_time \[ms\]:? ([\d.]+)")   # New format
    ]
    fwd_avg_time_patterns = [
        re.compile(r"average fwd time \[ms\]:? ([\d.]+)"),  # Old format
        re.compile(r"average_fwd_time \[ms\]:? ([\d.]+)")   # New format
    ]
    args_patterns = [
        re.compile(r"args = Namespace\((.+?)\)"),  # Original format
        re.compile(r"args=Namespace\((.+?)\)"),  # Original format
    ]
    first_token_latency_pattern = re.compile(r"1st token latency = ([\d.]+) ms")
    subsequent_token_latency_pattern = re.compile(r"2nd\+ token latency = ([\d.]+) ms")
    input_tokens_length_pattern = re.compile(r"input tokens length is (\d+)")
    output_token_nums_pattern = re.compile(r"output token nums = (\d+)")

    # Iterate over all log files in the directory
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    for log_file in log_files:
        # Remove "error_" prefix from the filename if it exists
        task_model = os.path.basename(log_file).replace(".log", "")
        if task_model.startswith("error_"):
            task_model = task_model[len("error_"):]

        # Split the filename into task and model
        try:
            task, model = task_model.split("_", 1)
        except ValueError:
            print(f"Skipping invalid log file name: {log_file}")
            continue

        with open(log_file, "r") as f:
            content = f.read()

        # Extract performance metrics
        total_time_match = total_time_pattern.search(content)
        pipeline_avg_time = extract_with_patterns(content, pipeline_avg_time_patterns)
        fwd_avg_time = extract_with_patterns(content, fwd_avg_time_patterns)
        args_match = extract_with_patterns(content, args_patterns)
        first_token_latency_match = first_token_latency_pattern.search(content)
        subsequent_token_latency_match = subsequent_token_latency_pattern.search(content)
        input_tokens_length_match = input_tokens_length_pattern.search(content)
        output_token_nums_match = output_token_nums_pattern.search(content)

        # Extract environment variables
        env_vars = extract_environment_variables(content)

        # Extract Python package versions
        packages = extract_python_packages(content)
        all_packages[model] = packages

        total_time = total_time_match.group(1) if total_time_match else None
        first_token_latency = first_token_latency_match.group(1) if first_token_latency_match else None
        subsequent_token_latency = subsequent_token_latency_match.group(1) if subsequent_token_latency_match else None
        input_tokens_length = input_tokens_length_match.group(1) if input_tokens_length_match else None
        output_token_nums = output_token_nums_match.group(1) if output_token_nums_match else None

        # Check if the log file indicates an error
        is_error = "error" in os.path.basename(log_file).lower()
        if is_error:
            # Extract content starting from the last occurrence of "Command Output:"
            command_output_index = content.rfind("Command Output:")
            if command_output_index != -1:
                error_logs[task_model] = content[command_output_index:].strip()
            else:
                error_logs[task_model] = content.strip()

        # Append performance data
        performance_data.append({
            "task": task,
            "model": model,
            "total_time": total_time,
            "pipeline_avg_time": pipeline_avg_time,
            "fwd_avg_time": fwd_avg_time,
            "first_token_latency": first_token_latency,
            "subsequent_token_latency": subsequent_token_latency,
            "input_tokens_length": input_tokens_length,
            "output_token_nums": output_token_nums,
            "args": args_match,
            "env_vars": env_vars,
            "packages": packages,
            "status": "error" if is_error else "success"
        })

    # Sort performance data by task name
    performance_data.sort(key=lambda x: x["task"])
    return performance_data, all_packages, error_logs


def parse_finetune_directory(log_dir):
    # Initialize data structures
    finetune_data = []
    error_logs = {}
    all_packages = {}

    # Regex pattern for extracting the "Executing command" section
    executing_command_pattern = re.compile(r"Executing command:\n(.*?)\nCommand Output:", re.DOTALL)

    # Iterate over all log files in the directory
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    for log_file in log_files:
        # Remove "error_" prefix from the filename if it exists
        task_model = os.path.basename(log_file).replace(".log", "")
        if task_model.startswith("error_"):
            task_model = task_model[len("error_"):]

        # Split the filename into task and model
        try:
            task, model = task_model.split("_", 1)
        except ValueError:
            print(f"Skipping invalid log file name: {log_file}")
            continue

        with open(log_file, "r") as f:
            content = f.read()

        # Extract environment variables
        env_vars = extract_environment_variables(content)

        # Extract Python package versions
        packages = extract_python_packages(content)
        all_packages[model] = packages

        # Extract the "Executing command" section
        executing_command_match = executing_command_pattern.search(content)
        executing_command = executing_command_match.group(1).strip() if executing_command_match else None

        # Check if the log file indicates an error
        is_error = "error" in os.path.basename(log_file).lower()
        if is_error:
            # Extract content starting from the last occurrence of "Command Output:"
            command_output_index = content.rfind("Command Output:")
            if command_output_index != -1:
                error_logs[task_model] = content[command_output_index:].strip()
            else:
                error_logs[task_model] = content.strip()

        # Find the output directory for the model
        output_dir = None
        output_dir_pattern = os.path.join(log_dir, f"{task}_{model.replace('/', '_')}_output")
        if os.path.isdir(output_dir_pattern):
            output_dir = output_dir_pattern

        # Append fine-tuning data
        finetune_data.append({
            "task": task,
            "model": model,
            "env_vars": env_vars,
            "packages": packages,
            "executing_command": executing_command,  # Add the extracted command
            "output_dir": output_dir,  # Add the output directory path
            "status": "error" if is_error else "success"
        })

    # Sort fine-tuning data by task name
    finetune_data.sort(key=lambda x: x["task"])
    return finetune_data, all_packages, error_logs