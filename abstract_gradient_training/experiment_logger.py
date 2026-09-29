import json
import os
import time
import uuid
import numpy as np

# Helper to serialize numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_experiment_log(experiment_name, hyperparameters, metrics, privacy_results, output_dir="experiment_logs"):
    """
    Saves experiment results to a JSON log file.

    Args:
        experiment_name (str): Name of the benchmark (e.g., 'sin_benchmark').
        hyperparameters (dict): Dictionary of hyperparameters (lr, batch_size, epochs, etc.).
        metrics (dict): Dictionary of scalar performance metrics (e.g., {'test_mse': 0.5}).
        privacy_results (dict): Dictionary containing privacy-accuracy trade-off data.
                                Expected format:
                                {
                                    'epsilon': [0.1, 1.0, ...],
                                    'global_sensitivity': {'mse': [], 'std': []},
                                    'smooth_sensitivity': {'mse': [], 'std': []}
                                }
        output_dir (str): Directory to save logs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{experiment_name}_{timestamp}_{unique_id}.json"
    filepath = os.path.join(output_dir, filename)

    log_entry = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "privacy_results": privacy_results
    }

    with open(filepath, 'w') as f:
        json.dump(log_entry, f, indent=4, cls=NpEncoder)

    print(f"[Logging] Experiment log saved to: {filepath}")
    return filepath
