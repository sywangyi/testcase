import os
import re
import glob
from html import escape
from datetime import datetime
from utils import parse_inference_directory, parse_finetune_directory

def parse_inference_logs(log_dir, output_html="performance_report.html", output_md="performance_report.md"):
    # Ensure output files are stored in the log_dir
    output_html = os.path.join(log_dir, output_html)
    output_md = os.path.join(log_dir, output_md)
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    performance_data, all_packages, error_logs = parse_inference_directory(log_dir)
    # Calculate statistics
    successful_count = sum(1 for data in performance_data if data["status"] == "success")
    failed_count = sum(1 for data in performance_data if data["status"] == "error")

    # Compare environment variables and package versions across models
    env_var_discrepancies = {}
    package_discrepancies = {}
    reference_env_vars = None
    reference_packages = None

    for model, env_vars in {data["model"]: data["env_vars"] for data in performance_data}.items():
        if reference_env_vars is None:
            reference_env_vars = env_vars
        else:
            for key, value in env_vars.items():
                if key not in reference_env_vars or reference_env_vars[key] != value:
                    env_var_discrepancies.setdefault(key, []).append((model, value))

    for model, packages in all_packages.items():
        if reference_packages is None:
            reference_packages = packages
        else:
            for package, version in packages.items():
                if package not in reference_packages or reference_packages[package] != version:
                    package_discrepancies.setdefault(package, []).append((model, version))

    # Generate HTML report
    with open(output_html, "w") as html_file:
        html_file.write(f"""
        <html>
        <head>
            <title>Performance Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f9;
                    color: #333;
                }}
                header {{
                    background-color: #0071C5;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                h1, h2 {{
                    color: #0071C5;
                }}
                table {{
                    width: 90%;
                    margin: 20px auto;
                    border-collapse: collapse;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
                th, td {{
                    padding: 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #0071C5;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f1f1f1;
                }}
                .status-success {{
                    color: #0071C5;
                    font-weight: bold;
                }}
                .status-error {{
                    color: red;
                    font-weight: bold;
                }}
                pre {{
                    background-color: #f4f4f9;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                footer {{
                    text-align: center;
                    padding: 15px;
                    background-color: #0071C5;
                    color: white;
                    margin-top: 20px; /* Add spacing above the footer */
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .discrepancy {{
                    color: red;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>Performance Report</h1>
            </header>
            <div class="container">
                <p><strong>Log Directory:</strong> {escape(log_dir)}</p>
                <p><strong>Report Generated At:</strong> {report_time}</p>
        """)

        # Add statistics
        html_file.write(f"<h2>Summary</h2>")
        html_file.write(f"<p><strong>Successful Models:</strong> {successful_count}</p>")
        html_file.write(f"<p><strong>Failed Models:</strong> {failed_count}</p>")

        # Performance table
        html_file.write("<h2>Task and Model Performance</h2>")
        html_file.write("<table>")
        html_file.write("<tr><th>Task</th><th>Model</th><th>Total Time (ms)</th><th>Pipeline Avg Time (ms)</th><th>Fwd Avg Time (ms)</th><th>1st Token Latency (ms)</th><th>2nd+ Token Latency (ms)</th><th>Input Tokens</th><th>Output Tokens</th><th>Status</th></tr>")
        for data in performance_data:
            status_class = "status-error" if data["status"] == "error" else "status-success"
            html_file.write(f"""
<tr>
    <td>{escape(data['task'])}</td>
    <td>{escape(data['model'])}</td>
    <td>{escape(str(data['total_time']) if data['total_time'] is not None else "N/A")}</td>
    <td>{escape(str(data['pipeline_avg_time']) if data['pipeline_avg_time'] is not None else "N/A")}</td>
    <td>{escape(str(data['fwd_avg_time']) if data['fwd_avg_time'] is not None else "N/A")}</td>
    <td>{escape(str(data['first_token_latency']) if data['first_token_latency'] is not None else "N/A")}</td>
    <td>{escape(str(data['subsequent_token_latency']) if data['subsequent_token_latency'] is not None else "N/A")}</td>
    <td>{escape(str(data['input_tokens_length']) if data['input_tokens_length'] is not None else "N/A")}</td>
    <td>{escape(str(data['output_token_nums']) if data['output_token_nums'] is not None else "N/A")}</td>
    <td class='{status_class}'>{escape(data['status'])}</td>
</tr>
""")
        html_file.write("</table>")

        # Environment Variables
        html_file.write("<h2>Environment Variables</h2>")
        html_file.write("<pre>")
        for key, value in reference_env_vars.items():
            html_file.write(f"{escape(key)}={escape(value)}\n")
        html_file.write("</pre>")
        if env_var_discrepancies:
            html_file.write("<h3>Discrepancies</h3>")
            for key, models in env_var_discrepancies.items():
                html_file.write(f"<h4>{escape(key)}</h4>")
                for model, value in models:
                    html_file.write(f"<p><strong>{escape(model)}:</strong> {escape(value)}</p>")

        # Python Packages
        html_file.write("<h2>Python Packages</h2>")
        html_file.write("<pre>")
        for package, version in reference_packages.items():
            html_file.write(f"{escape(package)}: {escape(version)}\n")
        html_file.write("</pre>")
        if package_discrepancies:
            html_file.write("<h3>Discrepancies</h3>")
            for package, models in package_discrepancies.items():
                html_file.write(f"<h4>{escape(package)}</h4>")
                for model, version in models:
                    html_file.write(f"<p><strong>{escape(model)}:</strong> {escape(version)}</p>")

        # Model arguments
        html_file.write("<h2>Model Arguments</h2>")
        for data in performance_data:
            html_file.write(f"<h3>{escape(data['model'])}</h3>")
            html_file.write(f"<pre>{escape(data['args'])}</pre>")

        # Error logs
        if error_logs:
            html_file.write("<h2>Error Logs</h2>")
            for task_model, log_content in error_logs.items():
                html_file.write(f"<h3>{escape(task_model)}</h3>")
                html_file.write(f"<pre>{escape(log_content)}</pre>")

        html_file.write("""
            </div>
            <footer>
                <p>Generated by Logs Parser</p>
            </footer>
        </body>
        </html>
        """)

    # Generate Markdown report
    with open(output_md, "w") as md_file:
        # Report header
        md_file.write(f"# Performance Report\n\n")
        md_file.write(f"**Log Directory:** {log_dir}\n\n")
        md_file.write(f"**Report Generated At:** {report_time}\n\n")

        # Summary
        md_file.write(f"## Summary\n")
        md_file.write(f"- **Successful Models:** {successful_count}\n")
        md_file.write(f"- **Failed Models:** {failed_count}\n\n")

        # Task and Model Performance
        md_file.write(f"## Task and Model Performance\n")
        md_file.write(f"| Task | Model | Total Time (ms) | Pipeline Avg Time (ms) | Fwd Avg Time (ms) | 1st Token Latency (ms) | 2nd+ Token Latency (ms) | Input Tokens | Output Tokens | Status |\n")
        md_file.write(f"|------|-------|-----------------|-------------------------|-------------------|-------------------------|--------------------------|--------------|---------------|--------|\n")
        for data in performance_data:
            status = "✅ Success" if data["status"] == "success" else "❌ Error"
            md_file.write(f"| {data['task']} | {data['model']} | {data['total_time'] or 'N/A'} | {data['pipeline_avg_time'] or 'N/A'} | {data['fwd_avg_time'] or 'N/A'} | {data['first_token_latency'] or 'N/A'} | {data['subsequent_token_latency'] or 'N/A'} | {data['input_tokens_length'] or 'N/A'} | {data['output_token_nums'] or 'N/A'} | {status} |\n")

        # Environment Variables
        md_file.write(f"\n## Environment Variables\n")
        md_file.write("```\n")
        for key, value in reference_env_vars.items():
            md_file.write(f"{key}={value}\n")
        md_file.write("```\n\n")
        if env_var_discrepancies:
            md_file.write(f"### Discrepancies\n")
            for key, models in env_var_discrepancies.items():
                md_file.write(f"#### {key}\n")
                for model, value in models:
                    md_file.write(f"- **{model}:** {value}\n")
            md_file.write("\n")

        # Python Packages
        md_file.write(f"## Python Packages\n")
        md_file.write("```\n")
        for package, version in reference_packages.items():
            md_file.write(f"{package}: {version}\n")
        md_file.write("```\n\n")
        if package_discrepancies:
            md_file.write(f"### Discrepancies\n")
            for package, models in package_discrepancies.items():
                md_file.write(f"#### {package}\n")
                for model, version in models:
                    md_file.write(f"- **{model}:** {version}\n")
            md_file.write("\n")

        # Model Arguments
        md_file.write(f"\n## Model Arguments\n")
        for data in performance_data:
            md_file.write(f"### {data['model']}\n")
            md_file.write(f"```\n{data['args']}\n```\n\n")

        # Error Logs
        if error_logs:
            md_file.write(f"## Error Logs\n")
            for task_model, log_content in error_logs.items():
                md_file.write(f"### {task_model}\n")
                md_file.write(f"```\n{log_content}\n```\n\n")

    print(f"Markdown report generated: {output_md}")
    print(f"HTML report generated: {output_html}")

def parse_finetune_logs(log_dir, output_html="finetune_report.html", output_md="finetune_report.md"):
    # Ensure output files are stored in the log_dir
    output_html = os.path.join(log_dir, output_html)
    output_md = os.path.join(log_dir, output_md)
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Parse the fine-tune directory
    finetune_data, all_packages, error_logs = parse_finetune_directory(log_dir)

    # Calculate statistics
    successful_count = sum(1 for data in finetune_data if data["status"] == "success")
    failed_count = sum(1 for data in finetune_data if data["status"] == "error")

    # Generate HTML report
    with open(output_html, "w") as html_file:
        html_file.write(f"""
        <html>
        <head>
            <title>Fine-Tune Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f9;
                    color: #333;
                }}
                header {{
                    background-color: #0071C5;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                h1, h2 {{
                    color: #0071C5;
                }}
                table {{
                    width: 90%;
                    margin: 20px auto;
                    border-collapse: collapse;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
                th, td {{
                    padding: 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #0071C5;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f1f1f1;
                }}
                .status-success {{
                    color: #0071C5;
                    font-weight: bold;
                }}
                .status-error {{
                    color: red;
                    font-weight: bold;
                }}
                pre {{
                    background-color: #f4f4f9;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                footer {{
                    text-align: center;
                    padding: 15px;
                    background-color: #0071C5;
                    color: white;
                    margin-top: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>Fine-Tune Report</h1>
            </header>
            <div class="container">
                <p><strong>Log Directory:</strong> {log_dir}</p>
                <p><strong>Report Generated At:</strong> {report_time}</p>
        """)

        # Add statistics
        html_file.write(f"<h2>Summary</h2>")
        html_file.write(f"<p><strong>Successful Models:</strong> {successful_count}</p>")
        html_file.write(f"<p><strong>Failed Models:</strong> {failed_count}</p>")

        # Fine-tune table
        html_file.write("<h2>Fine-Tune Performance</h2>")
        html_file.write("<table>")
        html_file.write("<tr><th>Task</th><th>Model</th><th>Status</th><th>Output Directory</th><th>Executing Command</th></tr>")
        for data in finetune_data:
            status_class = "status-error" if data["status"] == "error" else "status-success"
            html_file.write(f"""
<tr>
    <td>{data['task']}</td>
    <td>{data['model']}</td>
    <td class='{status_class}'>{data['status']}</td>
    <td>{data['output_dir'] or 'N/A'}</td>
    <td><pre>{data['executing_command'] or 'N/A'}</pre></td>
</tr>
""")
        html_file.write("</table>")

        # Error logs
        if error_logs:
            html_file.write("<h2>Error Logs</h2>")
            for task_model, log_content in error_logs.items():
                html_file.write(f"<h3>{task_model}</h3>")
                html_file.write(f"<pre>{log_content}</pre>")

        html_file.write("""
            </div>
            <footer>
                <p>Generated by Fine-Tune Logs Parser</p>
            </footer>
        </body>
        </html>
        """)

    # Generate Markdown report
    with open(output_md, "w") as md_file:
        # Report header
        md_file.write(f"# Fine-Tune Report\n\n")
        md_file.write(f"**Log Directory:** {log_dir}\n\n")
        md_file.write(f"**Report Generated At:** {report_time}\n\n")

        # Summary
        md_file.write(f"## Summary\n")
        md_file.write(f"- **Successful Models:** {successful_count}\n")
        md_file.write(f"- **Failed Models:** {failed_count}\n\n")

        # Fine-tune table
        md_file.write(f"## Fine-Tune Performance\n")
        md_file.write(f"| Task | Model | Status | Output Directory | Executing Command |\n")
        md_file.write(f"|------|-------|--------|------------------|-------------------|\n")
        for data in finetune_data:
            status = "✅ Success" if data["status"] == "success" else "❌ Error"
            md_file.write(f"| {data['task']} | {data['model']} | {status} | {data['output_dir'] or 'N/A'} | `{data['executing_command'] or 'N/A'}` |\n")

        # Error logs
        if error_logs:
            md_file.write(f"\n## Error Logs\n")
            for task_model, log_content in error_logs.items():
                md_file.write(f"### {task_model}\n")
                md_file.write(f"```\n{log_content}\n```\n\n")

    print(f"Markdown report generated: {output_md}")
    print(f"HTML report generated: {output_html}")


if __name__ == "__main__":
    import sys
    import os

    base_log_dir = sys.argv[1]
    
    # Process both inference and finetune directories using the same logic
    for prefix in ["inference", "finetune"]:
        # Find all directories that start with the prefix in the base log directory
        matching_dirs = [d for d in os.listdir(base_log_dir) 
                         if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith(prefix)]
        
        if matching_dirs:
            # Process each directory
            for dir_name in matching_dirs:
                log_dir = os.path.join(base_log_dir, dir_name)
                print(f"Processing {prefix} directory: {log_dir}")
                if prefix == "inference":
                    parse_inference_logs(log_dir, "inference_performance_report.html", "inference_performance_report.md")
                else:
                    # Assuming a similar function exists for finetune logs
                    parse_finetune_logs(log_dir, "finetune_performance_report.html", "finetune_performance_report.md")
        else:
            print(f"No {prefix} directories found in {base_log_dir}")