import json
import os


def clean_string(in_string):
    out_string = in_string.replace(" ", "_")
    out_string = out_string.replace(":", "_")
    out_string = out_string.replace(",", "_")
    out_string = out_string.replace("=", "_")
    out_string = out_string.replace(".", "_")
    out_string = out_string.replace("[", "")
    out_string = out_string.replace("]", "")
    out_string = out_string.replace("(", "")
    out_string = out_string.replace(")", "")
    out_string = out_string.replace("'", "")
    out_string = out_string.replace("%", "")
    return out_string


def title_to_filename(title, location, file_ending="png"):
    safe_title = clean_string(title)
    return f"{location}/{safe_title}.{file_ending}"


def save_to_file(plt, title, location):
    filename = title_to_filename(title, location=location)
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(fname=filename, bbox_inches="tight")
    plt.close()


def save_json_to_file(to_save_dict, file_name, location):
    file_location = f"{location}/{file_name}"
    with open(file_location, "w") as f:
        json.dump(to_save_dict, f, indent=2)
