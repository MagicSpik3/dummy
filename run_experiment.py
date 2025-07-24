# --- run_experiment.py ---
import toml
from orchestrator import ExperimentOrchestrator # Assuming the class is in this file

def main():
    """Main function to run the experiment."""
    # 1. Load the configuration
    with open("experiments.toml", "r", encoding="utf-8") as f:
        config = toml.load(f)

    # 2. Initialise the orchestrator
    orchestrator = ExperimentOrchestrator(config)

    # 3. Run the experiment
    results_dataframe = orchestrator.run_experiment()

    # 4. Save the results
    orchestrator.save_results(results_dataframe)
    
    print("\n--- First 5 rows of results ---")
    print(results_dataframe.head())

if __name__ == "__main__":
    main()
