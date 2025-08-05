import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your generator (assuming the code is in gen_v2.py)
from gen_v2 import EventStreamGenerator, Event

class GeneratorTester:
    """
    Comprehensive testing framework for the EventStreamGenerator.
    Performs pairwise evaluation of features and generates analysis.
    """
    
    def __init__(self):
        # Define the 5 main features to test
        self.features = [
            'temporal_dependency_strength',
            'long_term_dependency_strength', 
            'non_linear_dependency_strength',
            'out_of_order_strength',
            'fractal_behavior_strength'
        ]
        
        # Define the test values
        self.test_values = [0.0, 0.1, 0.5, 0.7, 1.0]
        
        # Default values for features not being tested
        self.default_values = {
            'temporal_dependency_strength': 0.7,
            'long_term_dependency_strength': 0.7,
            'non_linear_dependency_strength': 0.7,
            'out_of_order_strength': 0.3,
            'fractal_behavior_strength': 0.2,
            'concurrency_percentage': 0.5  # Keep this constant
        }
        
        # Results storage
        self.results = []
        self.experiment_counter = 0
        
    def generate_experiment_configs(self) -> List[Dict[str, Any]]:
        """Generate all pairwise combinations of features and values."""
        configs = []
        
        # Get all pairs of features
        feature_pairs = list(itertools.combinations(self.features, 2))
        print(f"Testing {len(feature_pairs)} feature pairs:")
        
        for i, (feat1, feat2) in enumerate(feature_pairs):
            print(f"  {i+1}. {feat1} vs {feat2}")
            
            # For each pair, test all combinations of values
            for val1 in self.test_values:
                for val2 in self.test_values:
                    config = self.default_values.copy()
                    config[feat1] = val1
                    config[feat2] = val2
                    config['experiment_id'] = len(configs) + 1
                    config['feature_pair'] = f"{feat1}_vs_{feat2}"
                    config['tested_features'] = [feat1, feat2]
                    config['tested_values'] = [val1, val2]
                    configs.append(config)
        
        print(f"\nTotal experiments to run: {len(configs)}")
        return configs
    
    def run_single_experiment(self, config: Dict[str, Any], max_events: int = 500) -> Dict[str, Any]:
        """Run a single experiment and collect metrics."""
        exp_id = config['experiment_id']
        
        # Create generator with the specified configuration
        generator = EventStreamGenerator(
            temporal_dependency_strength=config['temporal_dependency_strength'],
            long_term_dependency_strength=config['long_term_dependency_strength'],
            non_linear_dependency_strength=config['non_linear_dependency_strength'],
            out_of_order_strength=config['out_of_order_strength'],
            fractal_behavior_strength=config['fractal_behavior_strength'],
            concurrency_percentage=config['concurrency_percentage'],
            seed=42  # Fixed seed for reproducibility
        )
        
        # Generate events
        events_list = []
        try:
            for i, event in enumerate(generator.generate_event_stream(max_events=max_events)):
                events_list.append(event)
                if i >= max_events - 1:
                    break
        except Exception as e:
            print(f"Error in experiment {exp_id}: {e}")
            return None
        
        # Calculate metrics
        metrics = self.calculate_metrics(events_list, config)
        
        # Save CSV file
        filename = f"experiment_{exp_id:03d}_{config['feature_pair']}_" \
                  f"{str(config['tested_values'][0]).replace('.', ',')}_{str(config['tested_values'][1]).replace('.', ',')}.csv"
        
        # Create results directory if it doesn't exist
        os.makedirs('../../../../Desktop/Stream Enviorment/experiment_results', exist_ok=True)
        filepath = os.path.join('../../../../Desktop/Stream Enviorment/experiment_results', filename)
        
        generator.save_to_csv(events_list, filepath)
        metrics['csv_file'] = filepath
        
        return metrics
    
    def calculate_metrics(self, events: List[Event], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate various metrics from the generated events."""
        if not events:
            return {}
        
        metrics = {
            'experiment_id': config['experiment_id'],
            'feature_pair': config['feature_pair'],
            'tested_features': config['tested_features'],
            'tested_values': config['tested_values'],
            'total_events': len(events),
        }
        
        # Add configuration values
        for feat in self.features:
            metrics[f'config_{feat}'] = config[feat]
        metrics['config_concurrency_percentage'] = config['concurrency_percentage']
        
        # Basic event metrics
        start_events = [e for e in events if e.event_type.value == 'start']
        complete_events = [e for e in events if e.event_type.value == 'complete']
        
        metrics['start_events'] = len(start_events)
        metrics['complete_events'] = len(complete_events)
        metrics['start_complete_ratio'] = len(start_events) / len(complete_events) if complete_events else 0
        
        # Case metrics
        unique_cases = set(e.case_id for e in events)
        metrics['unique_cases'] = len(unique_cases)
        metrics['events_per_case'] = len(events) / len(unique_cases) if unique_cases else 0
        
        # Out-of-order metrics
        ooo_events = [e for e in events if e.attributes.get('out_of_order', False)]
        metrics['out_of_order_events'] = len(ooo_events)
        metrics['out_of_order_percentage'] = len(ooo_events) / len(events) * 100
        
        # Fractal metrics
        fractal_events = [e for e in events if 'fractal_depth' in e.attributes]
        metrics['fractal_events'] = len(fractal_events)
        metrics['fractal_percentage'] = len(fractal_events) / len(events) * 100
        
        # Temporal metrics
        timestamps = [e.timestamp for e in events]
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                         for i in range(len(timestamps)-1)]
            metrics['avg_time_between_events'] = np.mean(time_diffs)
            metrics['std_time_between_events'] = np.std(time_diffs)
        
        # Concurrency metrics
        concurrent_events = [e for e in events if e.attributes.get('concurrent_activities_in_case', 0) > 0]
        metrics['concurrent_events'] = len(concurrent_events)
        metrics['concurrency_percentage'] = len(concurrent_events) / len(events) * 100
        
        # Activity diversity
        unique_activities = set(e.activity for e in events)
        metrics['unique_activities'] = len(unique_activities)
        metrics['activity_diversity'] = len(unique_activities) / len(events)
        
        # Process diversity
        unique_processes = set(e.process_id for e in events)
        metrics['unique_processes'] = len(unique_processes)
        
        return metrics
    
    def run_all_experiments(self, max_events: int = 500):
        """Run all experiments and collect results."""
        configs = self.generate_experiment_configs()
        
        print(f"\nStarting {len(configs)} experiments...")
        print("This may take a while depending on max_events setting.")
        
        for i, config in enumerate(configs):
            if (i + 1) % 25 == 0:
                print(f"Completed {i + 1}/{len(configs)} experiments...")
            
            metrics = self.run_single_experiment(config, max_events)
            if metrics:
                self.results.append(metrics)
        
        print(f"\nCompleted all experiments. Successfully ran {len(self.results)} experiments.")
        
        # Save results to CSV
        results_df = pd.DataFrame(self.results)
        results_file = '../../../../Desktop/Stream Enviorment/experiment_results/all_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
        
        return results_df
    
    def create_heatmaps(self, results_df: pd.DataFrame, metrics_to_plot: List[str] = None):
        """Create heatmaps for each feature pair and metric."""
        if metrics_to_plot is None:
            # Default metrics of interest
            metrics_to_plot = [
                'out_of_order_percentage',
                'fractal_percentage', 
                'concurrency_percentage',
                'events_per_case',
                'activity_diversity'
            ]
        
        # Get all feature pairs
        feature_pairs = results_df['feature_pair'].unique()
        
        # Create heatmaps directory
        os.makedirs('../../../../Desktop/Stream Enviorment/experiment_results/heatmaps', exist_ok=True)
        
        for pair in feature_pairs:
            pair_data = results_df[results_df['feature_pair'] == pair]
            feat1, feat2 = pair.split('_vs_')
            
            # Create subplots for multiple metrics
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Heatmaps for {feat1} vs {feat2}', fontsize=16, fontweight='bold')
            
            for idx, metric in enumerate(metrics_to_plot):
                if idx >= 6:  # Only plot first 6 metrics
                    break
                    
                row, col = idx // 3, idx % 3
                ax = axes[row, col]
                
                # Create pivot table for heatmap
                pivot_data = pair_data.pivot_table(
                    values=metric,
                    index=f'config_{feat1}',
                    columns=f'config_{feat2}',
                    aggfunc='mean'
                )
                
                # Create heatmap
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt='.2f',
                    cmap='viridis',
                    ax=ax,
                    cbar_kws={'label': metric}
                )
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_xlabel(feat2.replace('_', ' ').title())
                ax.set_ylabel(feat1.replace('_', ' ').title())
            
            # Remove empty subplots
            if len(metrics_to_plot) < 6:
                for idx in range(len(metrics_to_plot), 6):
                    row, col = idx // 3, idx % 3
                    fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            
            # Save heatmap
            heatmap_file = f'experiment_results/heatmaps/heatmap_{pair}.png'
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved heatmap: {heatmap_file}")
    
    def generate_summary_report(self, results_df: pd.DataFrame):
        """Generate a comprehensive summary report."""
        report = []
        report.append("Event Stream Generator - Pairwise Testing Report")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total experiments: {len(results_df)}")
        report.append(f"Feature pairs tested: {len(results_df['feature_pair'].unique())}")
        report.append("")
        
        # Overall statistics
        report.append("Overall Statistics:")
        report.append("-" * 30)
        for col in ['total_events', 'out_of_order_percentage', 'fractal_percentage', 
                   'concurrency_percentage', 'events_per_case']:
            if col in results_df.columns:
                mean_val = results_df[col].mean()
                std_val = results_df[col].std()
                report.append(f"{col}: {mean_val:.2f} ± {std_val:.2f}")
        report.append("")
        
        # Feature pair analysis
        report.append("Feature Pair Analysis:")
        report.append("-" * 30)
        for pair in results_df['feature_pair'].unique():
            pair_data = results_df[results_df['feature_pair'] == pair]
            report.append(f"\n{pair}:")
            report.append(f"  Experiments: {len(pair_data)}")
            report.append(f"  Avg out-of-order: {pair_data['out_of_order_percentage'].mean():.2f}%")
            report.append(f"  Avg fractal: {pair_data['fractal_percentage'].mean():.2f}%")
            report.append(f"  Avg concurrency: {pair_data['concurrency_percentage'].mean():.2f}%")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = '../../../../Desktop/Stream Enviorment/experiment_results/summary_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Summary report saved to: {report_file}")
        return report_text

def main():
    """Main execution function."""
    print("Event Stream Generator - Comprehensive Pairwise Testing")
    print("=" * 60)
    
    # Create tester instance
    tester = GeneratorTester()
    
    # Ask user for number of events per experiment
    max_events = int(input("Enter max events per experiment (default 500): ") or "500")
    
    # Run all experiments
    results_df = tester.run_all_experiments(max_events=max_events)
    
    if len(results_df) > 0:
        # Create heatmaps
        print("\nGenerating heatmaps...")
        tester.create_heatmaps(results_df)
        
        # Generate summary report
        print("\nGenerating summary report...")
        tester.generate_summary_report(results_df)
        
        print("\nTesting complete!")
        print("Check the 'experiment_results' directory for:")
        print("- Individual CSV files for each experiment")
        print("- all_results.csv with aggregated metrics")
        print("- heatmaps/ directory with visualization")
        print("- summary_report.txt with analysis")
    else:
        print("No experiments completed successfully.")

if __name__ == "__main__":
    main()