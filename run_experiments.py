import os
import sys
import json
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

sys.path.append(os.path.abspath('./src'))
from QFed.qNN import QNNConfig, NoiseConfig
from QFed.qFL import FederatedConfig, QuantumFederatedLearning

def load_data():
    """Loads standardized datasets prior to experiment dispatch."""
    data_dir = Path('./dataset/processed_quantum')
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed subset not found at {data_dir}. Run preprocess.py first")
        
    train_data = torch.load(data_dir / 'train.pt')
    val_data = torch.load(data_dir / 'val.pt')
    test_data = torch.load(data_dir / 'test.pt')
    
    client_data = []
    # Assuming at least 4 clients based on standard prep
    num_clients = 4
    for i in range(1, num_clients + 1):
        try:
            client_data.append(torch.load(data_dir / f'client{i}.pt'))
        except FileNotFoundError:
            break
            
    return client_data, val_data, test_data

def run_single_experiment(experiment_config_dict):
    """
    Spins up a QuantumFederatedLearning node per spawned process.
    Config dictionary contains parameters mapped natively to the experiment grid.
    """
    run_name = experiment_config_dict['run_name']
    print(f"✅ Starting Experiment: {run_name}")
    
    # Map dictionary values to configs
    noise_config = NoiseConfig(
        depolarizing_p=experiment_config_dict.get('depolarizing', 0.0),
        amplitude_damping_gamma=experiment_config_dict.get('amplitude', 0.0)
    )
    
    qnn_config = QNNConfig(
        n_qubits=experiment_config_dict['qubits'],
        n_layers=experiment_config_dict['depth'],
        noise_config=noise_config
    )
    
    fed_config = FederatedConfig(
        num_clients=len(experiment_config_dict['client_data']),
        num_rounds=experiment_config_dict.get('rounds', 15),
        client_fraction=1.0, # Complete participation standard for testing
        dp_clip_norm=experiment_config_dict.get('dp_clip', 0.0),
        dp_noise_multiplier=experiment_config_dict.get('dp_noise', 0.0),
        use_secure_aggregation=experiment_config_dict.get('secure_agg', False),
        qnn_config=qnn_config,
        save_dir=f"./artifacts/runs/{run_name}"
    )
    
    client_data = experiment_config_dict.pop('client_data')
    val_data = experiment_config_dict.pop('val_data')
    test_data = experiment_config_dict.pop('test_data')
    
    system = QuantumFederatedLearning(fed_config)
    
    # Execute loop
    history = system.train(
        client_data_list=client_data,
        val_data=val_data,
        test_data=test_data
    )
    
    # Save completion dict
    result_path = Path(fed_config.save_dir) / 'experiment_results.json'
    with open(result_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"🏁 Finished Experiment: {run_name}")
    return run_name, history

def main():
    print("Loading datasets...")
    client_data, val_data, test_data = load_data()
    print(f"Loaded {len(client_data)} clients.")
    
    # Experiment Grid Dimensions
    qubits_list = [2, 4]
    depth_list = [1, 2]
    dp_noise_params = [(0.0, 0.0), (1.0, 0.5), (1.0, 1.0)] # (Clip, Sigma)
    noise_params = [(0.0, 0.0), (1e-3, 0.0)] # (Depolarizing, Amplitude Damping)
    
    experiments = []
    
    for (qubits, depth, dp, noise) in itertools.product(qubits_list, depth_list, dp_noise_params, noise_params):
        clip, sigma = dp
        depol, amp = noise
        secure_agg = True if clip > 0 else False
        
        run_name = f"q{qubits}_d{depth}_dp{sigma}_err{depol}"
        
        exp_dict = {
            'run_name': run_name,
            'qubits': qubits,
            'depth': depth,
            'rounds': 10,  # Shortened for concurrent execution example
            'dp_clip': clip,
            'dp_noise': sigma,
            'secure_agg': secure_agg,
            'depolarizing': depol,
            'amplitude': amp,
            'client_data': client_data,
            'val_data': val_data,
            'test_data': test_data
        }
        experiments.append(exp_dict)
        
    print(f"Total experiments configured: {len(experiments)}")
    print("Dispatching via multiprocessing...")
    
    # Execute concurrently
    max_workers = min(torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count(), len(experiments))
    max_workers = max(1, max_workers) # Ensure at least 1
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_experiment, exp): exp['run_name'] for exp in experiments}
        
        for future in as_completed(futures):
            rtn_name = futures[future]
            try:
                name, history = future.result()
                print(f"Completed processing for {name}. Final Acc: {history['test_acc'][-1]:.4f}")
            except Exception as e:
                print(f"Experiment {rtn_name} generated an exception: {e}")

if __name__ == "__main__":
    main()
