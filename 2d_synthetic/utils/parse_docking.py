import re
import json
import os

def parse_docking_scores(log_file):
    """
    Parses docking scores from a log file and calculates statistics.

    Args:
        log_file (str): Path to the .log file.

    Returns:
        dict: A dictionary containing docking scores for each index,
              the mean, min, and max docking score with their indices.
    """
    docking_energies = {}
    energies = []
    index_energy_map = {}

    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(r"Docking score for '(\d+)':\s+([-\d.]+)\s+kcal/mol", line)
                if match:
                    index = match.group(1)
                    energy = float(match.group(2))
                    index_energy_map[index] = energy
                    energies.append((index, energy))
    except FileNotFoundError:
        print(f"Error: File not found at {log_file}")
        return None

    # Store individual scores
    docking_energies.update(index_energy_map)

    if energies:
        values = [e[1] for e in energies]
        mean_energy = sum(values) / len(values)
        min_index, min_energy = min(energies, key=lambda x: x[1])
        max_index, max_energy = max(energies, key=lambda x: x[1])
        docking_energies['mean'] = mean_energy
        docking_energies['min'] = {'index': min_index, 'score': min_energy}
        docking_energies['max'] = {'index': max_index, 'score': max_energy}
    else:
        docking_energies['mean'] = None
        docking_energies['min'] = None
        docking_energies['max'] = None

    return docking_energies

def save_to_json(data, log_file_path):
    """
    Saves the docking score dictionary to a JSON file in the same directory as the log file.

    Args:
        data (dict): The dictionary of docking scores.
        log_file_path (str): Path to the original .log file.
    """
    log_dir = os.path.dirname(log_file_path)
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    output_file = os.path.join(log_dir, f"{base_name}.json")

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Docking scores saved to {output_file}")

if __name__ == "__main__":
    log_file_path = input("Please enter the path to your .log file: ").strip()
    docking_data = parse_docking_scores(log_file_path)
    if docking_data:
        save_to_json(docking_data, log_file_path)