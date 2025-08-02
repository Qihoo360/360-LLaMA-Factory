#!/usr/bin/env python3
"""
Needle in Haystack Evaluation Comparison Plotter

This script analyzes and compares needle haystack evaluation results across multiple experiments.
It can compare different models, configurations, data sources, and experimental runs.

Features:
- Compare performance across different models/configs
- Analyze performance by context length and needle position
- Generate heatmaps and line plots
- Support for multiple comparison modes
- Automatic detection of result directories
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class NIAHComparisonPlotter:
    """Main class for comparing and plotting needle haystack results."""
    
    def __init__(self, saves_dir: str = "saves"):
        self.saves_dir = Path(saves_dir)
        self.results_data = {}
        self.metadata = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def discover_results(self) -> Dict[str, Dict]:
        """Auto-discover all result directories and their data."""
        results = {}
        
        for result_dir in self.saves_dir.rglob("*"):
            if result_dir.is_dir():
                summary_file = result_dir / "summary.json"
                detailed_file = result_dir / "detailed_results.json"
                
                if summary_file.exists() and detailed_file.exists():
                    try:
                        # Load data
                        with open(summary_file) as f:
                            summary = json.load(f)
                        with open(detailed_file) as f:
                            detailed = json.load(f)
                        
                        # Extract metadata from path and data
                        rel_path = result_dir.relative_to(self.saves_dir)
                        experiment_name = str(rel_path).replace(os.sep, "_")
                        
                        # Parse metadata
                        metadata = self._extract_metadata(result_dir, summary, detailed)
                        
                        results[experiment_name] = {
                            'summary': summary,
                            'detailed': detailed,
                            'metadata': metadata,
                            'path': result_dir
                        }
                        
                        print(f"Found experiment: {experiment_name}")
                        print(f"  - Total examples: {summary['overall']['total_examples']}")
                        print(f"  - Average score: {summary['overall']['average_score']:.3f}")
                        print(f"  - Data source: {metadata.get('data_source', 'unknown')}")
                        print()
                        
                    except Exception as e:
                        print(f"Warning: Could not load results from {result_dir}: {e}")
        
        return results
    
    def _extract_metadata(self, result_dir: Path, summary: Dict, detailed: List) -> Dict:
        """Extract metadata from experiment results."""
        metadata = {}
        
        # Extract from path
        path_parts = str(result_dir).split(os.sep)
        metadata['model'] = path_parts[-2] if len(path_parts) >= 2 else 'unknown'
        metadata['experiment'] = path_parts[-1]
        
        # Extract from data
        if detailed:
            example = detailed[0]
            metadata['needle'] = example.get('needle', 'unknown')
            metadata['question'] = example.get('question', 'unknown')
            
            # Infer data source from needle text
            needle_text = example.get('needle', '').lower()
            if 'pg42demo' in needle_text or 'paulgraham' in str(result_dir).lower():
                metadata['data_source'] = 'paulgraham'
            elif 'custom' in str(result_dir).lower() or 'demo' in needle_text:
                metadata['data_source'] = 'custom'
            elif 'directory' in str(result_dir).lower():
                metadata['data_source'] = 'directory'
            else:
                metadata['data_source'] = 'custom'  # default
        
        # Extract context lengths and positions
        if summary.get('by_context_length'):
            metadata['context_lengths'] = sorted([int(k) for k in summary['by_context_length'].keys()])
        if summary.get('by_depth_percent'):
            metadata['depth_percents'] = sorted([float(k) for k in summary['by_depth_percent'].keys()])
        
        return metadata
    
    def plot_overall_comparison(self, experiments: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot overall performance comparison across experiments."""
        if experiments is None:
            experiments = list(self.results_data.keys())
        
        data = []
        for exp_name in experiments:
            if exp_name in self.results_data:
                result = self.results_data[exp_name]
                overall = result['summary']['overall']
                metadata = result['metadata']
                
                data.append({
                    'Experiment': exp_name.replace('_', '\n'),
                    'Average Score': overall['average_score'],
                    'Exact Match Rate': overall['exact_match_rate'],
                    'Total Examples': overall['total_examples'],
                    'Model': metadata['model'],
                    'Data Source': metadata['data_source']
                })
        
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Needle in Haystack - Overall Performance Comparison', fontsize=16, fontweight='bold')
        
        # Average Score comparison
        axes[0, 0].bar(range(len(df)), df['Average Score'], color=sns.color_palette("husl", len(df)))
        axes[0, 0].set_title('Average Score by Experiment')
        axes[0, 0].set_ylabel('Average Score')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(df['Experiment'], rotation=45, ha='right')
        axes[0, 0].set_ylim(0, 1.1)
        
        # Exact Match Rate comparison
        axes[0, 1].bar(range(len(df)), df['Exact Match Rate'], color=sns.color_palette("husl", len(df)))
        axes[0, 1].set_title('Exact Match Rate by Experiment')
        axes[0, 1].set_ylabel('Exact Match Rate')
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels(df['Experiment'], rotation=45, ha='right')
        axes[0, 1].set_ylim(0, 1.1)
        
        # Performance by Data Source
        if len(df['Data Source'].unique()) > 1:
            source_perf = df.groupby('Data Source')['Average Score'].mean()
            axes[1, 0].bar(source_perf.index, source_perf.values, color=sns.color_palette("Set2", len(source_perf)))
            axes[1, 0].set_title('Average Performance by Data Source')
            axes[1, 0].set_ylabel('Average Score')
            axes[1, 0].set_ylim(0, 1.1)
        else:
            axes[1, 0].text(0.5, 0.5, 'Only one data source\nacross experiments', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Data Source Analysis')
        
        # Total Examples
        axes[1, 1].bar(range(len(df)), df['Total Examples'], color=sns.color_palette("husl", len(df)))
        axes[1, 1].set_title('Number of Examples by Experiment')
        axes[1, 1].set_ylabel('Total Examples')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels(df['Experiment'], rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overall comparison saved to: {save_path}")
        else:
            plt.show()
    
    def plot_context_length_comparison(self, experiments: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot performance comparison by context length."""
        if experiments is None:
            experiments = list(self.results_data.keys())
        
        plt.figure(figsize=(12, 8))
        
        for exp_name in experiments:
            if exp_name in self.results_data:
                result = self.results_data[exp_name]
                by_length = result['summary']['by_context_length']
                
                lengths = sorted([int(k) for k in by_length.keys()])
                scores = [by_length[str(length)]['average_score'] for length in lengths]
                
                plt.plot(lengths, scores, marker='o', linewidth=2, markersize=8, 
                        label=exp_name.replace('_', ' '), alpha=0.8)
        
        plt.title('Performance by Context Length', fontsize=16, fontweight='bold')
        plt.xlabel('Context Length (tokens)', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1.1)
        
        # Add performance threshold lines
        plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
        plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% threshold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Context length comparison saved to: {save_path}")
        else:
            plt.show()
    
    def plot_depth_comparison(self, experiments: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot performance comparison by needle depth."""
        if experiments is None:
            experiments = list(self.results_data.keys())
        
        plt.figure(figsize=(12, 8))
        
        for exp_name in experiments:
            if exp_name in self.results_data:
                result = self.results_data[exp_name]
                by_depth = result['summary']['by_depth_percent']
                
                depths = sorted([float(k) for k in by_depth.keys()])
                scores = [by_depth[str(depth)]['average_score'] for depth in depths]
                
                plt.plot(depths, scores, marker='s', linewidth=2, markersize=8, 
                        label=exp_name.replace('_', ' '), alpha=0.8)
        
        plt.title('Performance by Needle Depth Position', fontsize=16, fontweight='bold')
        plt.xlabel('Needle Depth (% through context)', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1.1)
        
        # Highlight important positions
        plt.axvline(x=0, color='blue', linestyle=':', alpha=0.5, label='Beginning')
        plt.axvline(x=50, color='red', linestyle=':', alpha=0.5, label='Middle')
        plt.axvline(x=100, color='blue', linestyle=':', alpha=0.5, label='End')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Depth comparison saved to: {save_path}")
        else:
            plt.show()
    
    def plot_heatmap_comparison(self, experiment: str, save_path: Optional[str] = None):
        """Plot detailed heatmap for a specific experiment."""
        if experiment not in self.results_data:
            print(f"Experiment '{experiment}' not found!")
            return
        
        result = self.results_data[experiment]
        detailed = result['detailed']
        
        # Create matrix
        context_lengths = sorted(set(ex['context_length'] for ex in detailed))
        depth_percents = sorted(set(ex['depth_percent'] for ex in detailed))
        
        matrix = np.full((len(depth_percents), len(context_lengths)), np.nan)
        
        for example in detailed:
            depth_idx = depth_percents.index(example['depth_percent'])
            length_idx = context_lengths.index(example['context_length'])
            matrix[depth_idx, length_idx] = example['score']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, 
                   xticklabels=[f'{l:,}' for l in context_lengths],
                   yticklabels=[f'{d}%' for d in depth_percents],
                   annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                   cbar_kws={'label': 'Performance Score'})
        
        plt.title(f'Performance Heatmap: {experiment.replace("_", " ")}', fontsize=14, fontweight='bold')
        plt.xlabel('Context Length (tokens)', fontsize=12)
        plt.ylabel('Needle Depth (% through context)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        else:
            plt.show()
    
    def generate_comparison_report(self, experiments: Optional[List[str]] = None, output_dir: str = "comparison_plots"):
        """Generate a comprehensive comparison report with all plots."""
        if experiments is None:
            experiments = list(self.results_data.keys())
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating comparison report for {len(experiments)} experiments...")
        print(f"Output directory: {output_dir}")
        
        # Overall comparison
        self.plot_overall_comparison(experiments, os.path.join(output_dir, "overall_comparison.png"))
        
        # Context length comparison
        self.plot_context_length_comparison(experiments, os.path.join(output_dir, "context_length_comparison.png"))
        
        # Depth comparison
        self.plot_depth_comparison(experiments, os.path.join(output_dir, "depth_comparison.png"))
        
        # Individual heatmaps
        for exp in experiments:
            if exp in self.results_data:
                heatmap_path = os.path.join(output_dir, f"heatmap_{exp}.png")
                self.plot_heatmap_comparison(exp, heatmap_path)
        
        # Generate summary report
        self._generate_text_report(experiments, os.path.join(output_dir, "comparison_report.txt"))
        
        print(f"\nComparison report generated successfully!")
        print(f"Files created in '{output_dir}':")
        for file in sorted(os.listdir(output_dir)):
            print(f"  - {file}")
    
    def _generate_text_report(self, experiments: List[str], output_path: str):
        """Generate a text summary report."""
        with open(output_path, 'w') as f:
            f.write("Needle in Haystack Evaluation Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Experiments: {len(experiments)}\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("Experiment Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Experiment':<30} {'Avg Score':<10} {'Exact Match':<12} {'Examples':<10} {'Data Source':<15}\n")
            f.write("-" * 80 + "\n")
            
            for exp in experiments:
                if exp in self.results_data:
                    result = self.results_data[exp]
                    overall = result['summary']['overall']
                    metadata = result['metadata']
                    
                    f.write(f"{exp[:29]:<30} {overall['average_score']:<10.3f} "
                           f"{overall['exact_match_rate']:<12.3f} {overall['total_examples']:<10} "
                           f"{metadata.get('data_source', 'unknown'):<15}\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
            
            # Best performing experiments
            f.write("Performance Rankings:\n")
            f.write("-" * 30 + "\n")
            
            scores = []
            for exp in experiments:
                if exp in self.results_data:
                    score = self.results_data[exp]['summary']['overall']['average_score']
                    scores.append((exp, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (exp, score) in enumerate(scores[:5], 1):
                f.write(f"{i}. {exp}: {score:.3f}\n")
        
        print(f"Text report saved to: {output_path}")
    
    def load_results(self):
        """Load all discovered results."""
        print("Discovering needle haystack evaluation results...")
        self.results_data = self.discover_results()
        print(f"Found {len(self.results_data)} experiments.")
        return self.results_data


def main():
    parser = argparse.ArgumentParser(description="Compare Needle in Haystack evaluation results")
    parser.add_argument("--saves-dir", default="saves", help="Directory containing saved results")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to compare")
    parser.add_argument("--output-dir", default="comparison_plots", help="Output directory for plots")
    parser.add_argument("--plot-type", choices=["overall", "context", "depth", "heatmap", "all"], 
                       default="all", help="Type of plot to generate")
    parser.add_argument("--experiment-for-heatmap", help="Specific experiment for heatmap (required if plot-type=heatmap)")
    
    args = parser.parse_args()
    
    # Create plotter and load results
    plotter = NIAHComparisonPlotter(args.saves_dir)
    plotter.load_results()
    
    if not plotter.results_data:
        print("No needle haystack results found!")
        return
    
    # Generate requested plots
    if args.plot_type == "all":
        plotter.generate_comparison_report(args.experiments, args.output_dir)
    elif args.plot_type == "overall":
        plotter.plot_overall_comparison(args.experiments)
    elif args.plot_type == "context":
        plotter.plot_context_length_comparison(args.experiments)
    elif args.plot_type == "depth":
        plotter.plot_depth_comparison(args.experiments)
    elif args.plot_type == "heatmap":
        if not args.experiment_for_heatmap:
            print("--experiment-for-heatmap is required when plot-type=heatmap")
            return
        plotter.plot_heatmap_comparison(args.experiment_for_heatmap)


if __name__ == "__main__":
    main()