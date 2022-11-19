import json
import os

import numpy as np


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


def save_json_to_file(to_save_dict, file_name, location, clean_dict=False):
    if clean_dict:
        for item, value in to_save_dict.items():
            if isinstance(value, np.ndarray):
                to_save_dict[item] = value.tolist()
            elif isinstance(value, np.int64):
                to_save_dict[item] = int(value)
    file_location = f"{location}/{file_name}"
    with open(file_location, "w") as f:
        json.dump(to_save_dict, f, indent=2)
