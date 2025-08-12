import fire


def main(file_names, out_name):
    info_list = [
        "INFO:root:args",
        "INFO:root:total",
        "INFO:root:pipeline",
        "INFO:root:average_fwd_time",
        "INFO:root:model",
        "INFO:root:1st",
        "INFO:root:2nd+",
        "INFO:root:similarity",
    ]
    write_list = []
    file_names = file_names.split(",")
    for file_name in file_names:
        file = open(file_name, "r")
        file_list = file.readlines()
        for val in file_list:
            if val.startswith("test"):
                write_list.append("-" * 200)
                write_list.append("\n\n")
                write_list.append(val)
            for info in info_list:
                if val.startswith(info):
                    write_list.append(val)

    write_file = open(out_name, "w", encoding="utf-8")
    write_file.write("You can find task name after the ------------\n")
    write_file.write(
        "Model name and configs are in the INFO:root:args=Namespace(...)\n"
    )
    write_file.write(
        "Note that args in [batch_size, num_beam, input_tokens, output_tokens] only works for text-generation and summarization\n"
    )
    write_file.write(
        "We also have model forward time in the log because the pipeline contains process and model forward. The process time may have a large portion of the pipeline\n"
    )
    write_file.write("\n\n\n")
    for write_info in write_list:
        write_file.write(write_info)
        write_file.write("\n")


if __name__ == "__main__":
    fire.Fire(main)
