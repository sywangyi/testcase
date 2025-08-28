import os
import re
import glob
from html import escape
from datetime import datetime
from utils import parse_inference_directory

# Helper function to check if both values are None
def is_both_none(value1, value2):
    return value1 is None and value2 is None

def string_to_float_avg(total_time):
    # Process total_time as a list of floats
    if total_time is None or total_time == "N/A":
        return None
    try:
        total_time_values = [float(x.strip()) for x in total_time.split(",")]
        return sum(total_time_values) / len(total_time_values)  # Calculate average
    except ValueError:
        return None


def parse_logs(log_dirs, output_html="performance_comparison_report.html"):
    # Generate a timestamp for the report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_html = f"performance_comparison_report_{timestamp}.html"
    output_md = f"performance_comparison_report_{timestamp}.md"

    # Initialize data structures
    performance_data_list = []
    report_generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Generate the report time once

    # Parse all directories
    for log_dir in log_dirs:
        performance_data, _, _ = parse_inference_directory(log_dir)
        performance_data_list.append(performance_data)

    # Combine data for comparison
    combined_data = {}
    if len(log_dirs) > 1:
        for data1 in performance_data_list[0]:
            for data2 in performance_data_list[1]:
                if data1["task"] == data2["task"] and data1["model"] == data2["model"]:
                    task = data1["task"]
                    if task not in combined_data:
                        combined_data[task] = []
                    combined_data[task].append({
                        "model": data1["model"],
                        "total_time_1": string_to_float_avg(data1["total_time"]),
                        "total_time_2": string_to_float_avg(data2["total_time"]),
                        "pipeline_avg_time_1": data1["pipeline_avg_time"],
                        "pipeline_avg_time_2": data2["pipeline_avg_time"],
                        "fwd_avg_time_1": data1["fwd_avg_time"],
                        "fwd_avg_time_2": data2["fwd_avg_time"],
                        "first_token_latency_1": data1["first_token_latency"],
                        "first_token_latency_2": data2["first_token_latency"],
                        "subsequent_token_latency_1": data1["subsequent_token_latency"],
                        "subsequent_token_latency_2": data2["subsequent_token_latency"],
                        "args_1": data1["args"],
                        "args_2": data2["args"],
                        "env_vars_1": data1["env_vars"],
                        "env_vars_2": data2["env_vars"],
                        "packages_1": data2["packages"],
                        "packages_2": data2["packages"],
                    })

    # Sort combined_data by task name (case-insensitive) and model name (case-insensitive)
    sorted_combined_data = {
        task: sorted(data_list, key=lambda x: x["model"].lower())  # Sort models by lowercase names
        for task, data_list in sorted(combined_data.items(), key=lambda x: x[0].lower())  # Sort tasks by lowercase names
    }

    # Generate HTML report
    with open(output_html, "w") as html_file:
        html_file.write(f"""
        <html>
        <head>
            <title>Performance Comparison Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f9;
                    color: #333;
                }}
                table {{
                    width: 90%;
                    margin: 20px auto;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #0071C5;
                    color: white;
                }}
                .improved {{
                    color: green;
                    font-weight: bold;
                }}
                .deteriorated {{
                    color: red;
                    font-weight: bold;
                }}
                .hidden {{
                    display: none;
                 }}
            </style>
            <script>
                function toggleArgs() {{
                    const elements = document.querySelectorAll('.args-row');
                    elements.forEach(el => {{
                        el.classList.toggle('hidden');
                    }});
                }}
            </script>
        </head>
        <body>
            <h1>Performance Comparison Report</h1>
            <p><strong>Log Directories:</strong><br>{'<br>'.join(map(escape, log_dirs))}</p>
            <p><strong>Report Generated On:</strong> {report_generated_time}</p>
            <button onclick="toggleArgs()">Show/Hide Args and Envs</button>
        """)

        env_vars_filter = ["LD_LIBRARY_PATH", "PATH", "LS_COLORS"]  # Add environment variables you want to exclude

        for task, data_list in sorted_combined_data.items():
            html_file.write(f"""
            <h2>Task: {escape(task)}</h2>
            <table>
                <tr>
                    <th>Model</th>
            """)

            # Dynamically generate table headers based on non-None columns
            show_total_time = not is_both_none(data_list[0]["total_time_1"], data_list[0]["total_time_2"])
            show_pipeline_avg_time = not is_both_none(data_list[0]["pipeline_avg_time_1"], data_list[0]["pipeline_avg_time_2"])
            show_fwd_avg_time = not is_both_none(data_list[0]["fwd_avg_time_1"], data_list[0]["fwd_avg_time_2"])
            show_first_token_latency = not is_both_none(data_list[0]["first_token_latency_1"], data_list[0]["first_token_latency_2"])
            show_subsequent_token_latency = not is_both_none(data_list[0]["subsequent_token_latency_1"], data_list[0]["subsequent_token_latency_2"])
            show_args = not is_both_none(data_list[0]["args_1"], data_list[0]["args_2"])
            show_envs = not is_both_none(data_list[0]["env_vars_1"], data_list[0]["env_vars_2"])
            show_packages = not is_both_none(data_list[0]["packages_1"], data_list[0]["packages_2"])

            if show_total_time:
                html_file.write("<th>Total Time (Dir 1)</th><th>Total Time (Dir 2)</th>")
            if show_pipeline_avg_time:
                html_file.write("<th>Pipeline Avg Time (Dir 1)</th><th>Pipeline Avg Time (Dir 2)</th>")
            if show_fwd_avg_time:
                html_file.write("<th>Fwd Avg Time (Dir 1)</th><th>Fwd Avg Time (Dir 2)</th>")
            if show_first_token_latency:
                html_file.write("<th>1st Token Latency (Dir 1)</th><th>1st Token Latency (Dir 2)</th>")
            if show_subsequent_token_latency:
                html_file.write("<th>2nd+ Token Latency (Dir 1)</th><th>2nd+ Token Latency (Dir 2)</th>")
            if show_args:
                html_file.write("<th class='args-row'>Args (Dir 1)</th><th class='args-row'>Args (Dir 2)</th>")
            if show_envs:
                html_file.write("<th class='args-row'>Environment Variables (Dir 1)</th><th class='args-row'>Environment Variables (Dir 2)</th>")
            if show_packages:
                html_file.write("<th class='args-row'>Python Packages (Dir 1)</th><th class='args-row'>Python Packages (Dir 2)</th>")

            html_file.write("</tr>")

            for data in data_list:
                html_file.write("<tr>")
                html_file.write(f"<td>{escape(data['model'])}</td>")

                # Dynamically generate table rows based on non-None columns
                if show_total_time:
                    html_file.write(f"<td>{escape(str(data['total_time_1']) if data['total_time_1'] is not None else 'N/A')}</td>")
                    html_file.write(f"<td>{escape(str(data['total_time_2']) if data['total_time_2'] is not None else 'N/A')}</td>")
                if show_pipeline_avg_time:
                    html_file.write(f"<td>{escape(str(data['pipeline_avg_time_1']) if data['pipeline_avg_time_1'] is not None else 'N/A')}</td>")
                    html_file.write(f"<td>{escape(str(data['pipeline_avg_time_2']) if data['pipeline_avg_time_2'] is not None else 'N/A')}</td>")
                if show_fwd_avg_time:
                    html_file.write(f"<td>{escape(str(data['fwd_avg_time_1']) if data['fwd_avg_time_1'] is not None else 'N/A')}</td>")
                    html_file.write(f"<td>{escape(str(data['fwd_avg_time_2']) if data['fwd_avg_time_2'] is not None else 'N/A')}</td>")
                if show_first_token_latency:
                    html_file.write(f"<td>{escape(str(data['first_token_latency_1']) if data['first_token_latency_1'] is not None else 'N/A')}</td>")
                    html_file.write(f"<td>{escape(str(data['first_token_latency_2']) if data['first_token_latency_2'] is not None else 'N/A')}</td>")
                if show_subsequent_token_latency:
                    html_file.write(f"<td>{escape(str(data['subsequent_token_latency_1']) if data['subsequent_token_latency_1'] is not None else 'N/A')}</td>")
                    html_file.write(f"<td>{escape(str(data['subsequent_token_latency_2']) if data['subsequent_token_latency_2'] is not None else 'N/A')}</td>")
                if show_args:
                    html_file.write(f"<td class='args-row'>{escape(data['args_1'])}</td>")
                    html_file.write(f"<td class='args-row'>{escape(data['args_2'])}</td>")
                if show_envs:
                    html_file.write(f"<td class='args-row'><pre>{escape(format_env_vars(data['env_vars_1']))}</pre></td>")  # Format env_vars_1 with filter
                    html_file.write(f"<td class='args-row'><pre>{escape(format_env_vars(data['env_vars_2']))}</pre></td>")  # Format env_vars_2 with filter
                if show_packages:
                    html_file.write(f"<td class='args-row'><pre>{escape(format_env_vars(data['packages_1']))}</pre></td>")
                    html_file.write(f"<td class='args-row'><pre>{escape(format_env_vars(data['packages_2']))}</pre></td>")

                html_file.write("</tr>")

            # Add summary row
            html_file.write("<tr><th>Summary</th>")
            if show_total_time:
                html_file.write("<th colspan='2'>Total Time Change</th>")
            if show_pipeline_avg_time:
                html_file.write("<th colspan='2'>Pipeline Avg Time Change</th>")
            if show_fwd_avg_time:
                html_file.write("<th colspan='2'>Fwd Avg Time Change</th>")
            if show_first_token_latency:
                html_file.write("<th colspan='2'>1st Token Latency Change</th>")
            if show_subsequent_token_latency:
                html_file.write("<th colspan='2'>2nd+ Token Latency Change</th>")
            html_file.write("</tr>")

            for data in data_list:
                html_file.write("<tr>")
                html_file.write(f"<td>{escape(data['model'])}</td>")
                if show_total_time:
                    html_file.write(f"<td colspan='2' class='{get_change_class(data['total_time_1'], data['total_time_2'])}'>"
                                    f"{calculate_change(data['total_time_1'], data['total_time_2'])}</td>")
                if show_pipeline_avg_time:
                    html_file.write(f"<td colspan='2' class='{get_change_class(data['pipeline_avg_time_1'], data['pipeline_avg_time_2'])}'>"
                                    f"{calculate_change(data['pipeline_avg_time_1'], data['pipeline_avg_time_2'])}</td>")
                if show_fwd_avg_time:
                    html_file.write(f"<td colspan='2' class='{get_change_class(data['fwd_avg_time_1'], data['fwd_avg_time_2'])}'>"
                                    f"{calculate_change(data['fwd_avg_time_1'], data['fwd_avg_time_2'])}</td>")
                if show_first_token_latency:
                    html_file.write(f"<td colspan='2' class='{get_change_class(data['first_token_latency_1'], data['first_token_latency_2'])}'>"
                                    f"{calculate_change(data['first_token_latency_1'], data['first_token_latency_2'])}</td>")
                if show_subsequent_token_latency:
                    html_file.write(f"<td colspan='2' class='{get_change_class(data['subsequent_token_latency_1'], data['subsequent_token_latency_2'])}'>"
                                    f"{calculate_change(data['subsequent_token_latency_1'], data['subsequent_token_latency_2'])}</td>")
                html_file.write("</tr>")

            html_file.write("</table>")

        html_file.write("""
        </body>
        </html>
        """)

    print(f"HTML report generated: {output_html}")

    generate_markdown_report(sorted_combined_data, log_dirs, report_generated_time, output_md=output_md)

def calculate_change(value1, value2):
    """
    Calculate the percentage change between two values.
    A negative percentage indicates improvement (smaller is better).
    """
    if value1 is None or value2 is None:
        return "N/A"
    try:
        value1 = float(value1)
        value2 = float(value2)
        change = ((value1 - value2) / value1) * 100  # Reverse the calculation to reflect "smaller is better"
        return f"{change:.2f}%"
    except ZeroDivisionError:
        return "N/A"


def get_change_class(value1, value2):
    """
    Determine the CSS class for the change.
    'improved' if the value decreased (smaller is better), 'deteriorated' otherwise.
    """
    if value1 is None or value2 is None:
        return ""
    value1 = float(value1)
    value2 = float(value2)
    try:
        change = ((value1 - value2) / value1) * 100  # Reverse the calculation to reflect "smaller is better"
        return "improved" if change > 0 else "deteriorated"
    except ZeroDivisionError:
        return ""


def format_env_vars(env_vars):
    """
    Format the environment variables dictionary into a string with each key=value on a new line.
    Optionally filter out variables based on the env_vars_filter list.
    """
    if not env_vars:
        return "N/A"
    
    env_vars = {key: value for key, value in env_vars.items()}
    return "\n".join(f"{key}={value}" for key, value in env_vars.items())


def generate_markdown_report(sorted_combined_data, log_dirs, report_generated_time, output_md="performance_comparison_report.md"):
    """
    Generate a Markdown report for performance comparison, excluding columns where both values are None.
    """
    with open(output_md, "w") as md_file:
        # Write the header
        md_file.write(f"# Performance Comparison Report\n\n")
        md_file.write(f"**Report Generated On:** {report_generated_time}\n\n")
        md_file.write(f"**Log Directories:**\n\n")
        for log_dir in log_dirs:
            md_file.write(f"- {log_dir}\n")
        md_file.write("\n")

        # Iterate through tasks and generate tables
        for task, data_list in sorted_combined_data.items():
            md_file.write(f"## Task: {task}\n\n")

            # Determine which columns to show based on the first data entry
            show_total_time = not is_both_none(data_list[0]["total_time_1"], data_list[0]["total_time_2"])
            show_pipeline_avg_time = not is_both_none(data_list[0]["pipeline_avg_time_1"], data_list[0]["pipeline_avg_time_2"])
            show_fwd_avg_time = not is_both_none(data_list[0]["fwd_avg_time_1"], data_list[0]["fwd_avg_time_2"])
            show_first_token_latency = not is_both_none(data_list[0]["first_token_latency_1"], data_list[0]["first_token_latency_2"])
            show_subsequent_token_latency = not is_both_none(data_list[0]["subsequent_token_latency_1"], data_list[0]["subsequent_token_latency_2"])

            # Write the table header dynamically
            md_file.write("| Model ")
            if show_total_time:
                md_file.write("| Total Time (Dir 1) | Total Time (Dir 2) | Total Time Change ")
            if show_pipeline_avg_time:
                md_file.write("| Pipeline Avg Time (Dir 1) | Pipeline Avg Time (Dir 2) | Pipeline Avg Time Change ")
            if show_fwd_avg_time:
                md_file.write("| Fwd Avg Time (Dir 1) | Fwd Avg Time (Dir 2) | Fwd Avg Time Change ")
            if show_first_token_latency:
                md_file.write("| 1st Token Latency (Dir 1) | 1st Token Latency (Dir 2) | 1st Token Latency Change ")
            if show_subsequent_token_latency:
                md_file.write("| 2nd+ Token Latency (Dir 1) | 2nd+ Token Latency (Dir 2) | 2nd+ Token Latency Change ")
            md_file.write("|\n")

            # Write the table separator dynamically
            md_file.write("|-------")
            if show_total_time:
                md_file.write("|--------------------|--------------------|-------------------")
            if show_pipeline_avg_time:
                md_file.write("|--------------------------|--------------------------|--------------------------")
            if show_fwd_avg_time:
                md_file.write("|---------------------|---------------------|---------------------")
            if show_first_token_latency:
                md_file.write("|--------------------------|--------------------------|--------------------------")
            if show_subsequent_token_latency:
                md_file.write("|---------------------------|---------------------------|---------------------------")
            md_file.write("|\n")

            # Write the table rows dynamically
            for data in data_list:
                md_file.write(f"| {data['model']} ")
                if show_total_time:
                    total_time_change = calculate_change(data['total_time_1'], data['total_time_2']) if data['total_time_1'] and data['total_time_2'] else "N/A"
                    md_file.write(f"| {data['total_time_1'] if data['total_time_1'] is not None else 'N/A'} ")
                    md_file.write(f"| {data['total_time_2'] if data['total_time_2'] is not None else 'N/A'} ")
                    md_file.write(f"| {total_time_change} ")
                if show_pipeline_avg_time:
                    pipeline_avg_time_change = calculate_change(data['pipeline_avg_time_1'], data['pipeline_avg_time_2']) if data['pipeline_avg_time_1'] and data['pipeline_avg_time_2'] else "N/A"
                    md_file.write(f"| {data['pipeline_avg_time_1'] if data['pipeline_avg_time_1'] is not None else 'N/A'} ")
                    md_file.write(f"| {data['pipeline_avg_time_2'] if data['pipeline_avg_time_2'] is not None else 'N/A'} ")
                    md_file.write(f"| {pipeline_avg_time_change} ")
                if show_fwd_avg_time:
                    fwd_avg_time_change = calculate_change(data['fwd_avg_time_1'], data['fwd_avg_time_2']) if data['fwd_avg_time_1'] and data['fwd_avg_time_2'] else "N/A"
                    md_file.write(f"| {data['fwd_avg_time_1'] if data['fwd_avg_time_1'] is not None else 'N/A'} ")
                    md_file.write(f"| {data['fwd_avg_time_2'] if data['fwd_avg_time_2'] is not None else 'N/A'} ")
                    md_file.write(f"| {fwd_avg_time_change} ")
                if show_first_token_latency:
                    first_token_latency_change = calculate_change(data['first_token_latency_1'], data['first_token_latency_2']) if data['first_token_latency_1'] and data['first_token_latency_2'] else "N/A"
                    md_file.write(f"| {data['first_token_latency_1'] if data['first_token_latency_1'] is not None else 'N/A'} ")
                    md_file.write(f"| {data['first_token_latency_2'] if data['first_token_latency_2'] is not None else 'N/A'} ")
                    md_file.write(f"| {first_token_latency_change} ")
                if show_subsequent_token_latency:
                    subsequent_token_latency_change = calculate_change(data['subsequent_token_latency_1'], data['subsequent_token_latency_2']) if data['subsequent_token_latency_1'] and data['subsequent_token_latency_2'] else "N/A"
                    md_file.write(f"| {data['subsequent_token_latency_1'] if data['subsequent_token_latency_1'] is not None else 'N/A'} ")
                    md_file.write(f"| {data['subsequent_token_latency_2'] if data['subsequent_token_latency_2'] is not None else 'N/A'} ")
                    md_file.write(f"| {subsequent_token_latency_change} ")
                md_file.write("|\n")

            md_file.write("\n")

    print(f"Markdown report generated: {output_md}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python logs_compare.py <log_dir1> [<log_dir2> ...]")
        sys.exit(1)

    log_dirs = sys.argv[1:]
    parse_logs(log_dirs)