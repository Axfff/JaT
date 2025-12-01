#!/bin/bash

# Script to compare all experiments and generate visualizations
# This script automates the comparison of predicted spectrums with groundtruth

echo "Starting spectrum comparison for all experiments..."
echo "=================================================="

# Create output directory
mkdir -p spectrum_comparisons

# Define experiments
# Format: experiment_name:checkpoint_path:dataset_mode:pred_mode

experiments=(
    "main_raw_x:checkpoints/model_main_raw_x_ep50.pth:raw:x"
    "baseline_raw_eps:checkpoints/model_baseline_raw_eps_ep50.pth:raw:epsilon"
    "control_spec_eps:checkpoints/model_control_spec_eps_ep50.pth:spectrogram:epsilon"
    "test_spec_x:checkpoints/model_test_spec_x_ep50.pth:spectrogram:x"
)

# Alternatively, use pre-generated results directories
# Format: experiment_name:results_dir:dataset_mode

use_pregenerated=true

if [ "$use_pregenerated" = true ]; then
    echo "Using pre-generated results from results/ directory"
    
    # Check if results directory exists
    if [ ! -d "results" ]; then
        echo "Error: results directory not found"
        exit 1
    fi
    
    # Process each experiment result directory
    for exp_dir in results/*/; do
        exp_name=$(basename "$exp_dir")
        echo ""
        echo "Processing: $exp_name"
        echo "-------------------"
        
        # Determine dataset mode from experiment name
        if [[ $exp_name == *"spec"* ]]; then
            dataset_mode="spectrogram"
        else
            dataset_mode="raw"
        fi
        
        output_dir="spectrum_comparisons/${exp_name}"
        
        python visualize_spectrum.py \
            --use_predictions "$exp_dir" \
            --data_root ./data \
            --dataset_mode "$dataset_mode" \
            --num_samples 16 \
            --num_display 8 \
            --output_dir "$output_dir" \
            --subset testing
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $exp_name"
        else
            echo "✗ Failed: $exp_name"
        fi
    done
else
    echo "Generating predictions from checkpoints..."
    
    for exp in "${experiments[@]}"; do
        IFS=':' read -r exp_name checkpoint dataset_mode pred_mode <<< "$exp"
        
        echo ""
        echo "Processing: $exp_name"
        echo "-------------------"
        
        output_dir="spectrum_comparisons/${exp_name}"
        
        python visualize_spectrum.py \
            --checkpoint "$checkpoint" \
            --data_root ./data \
            --dataset_mode "$dataset_mode" \
            --pred_mode "$pred_mode" \
            --num_samples 16 \
            --num_display 8 \
            --output_dir "$output_dir" \
            --subset testing
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $exp_name"
        else
            echo "✗ Failed: $exp_name"
        fi
    done
fi

# Generate summary report
echo ""
echo "=================================================="
echo "Generating summary report..."
echo "=================================================="

summary_file="spectrum_comparisons/SUMMARY.md"

cat > "$summary_file" << 'EOF'
# Spectrum Comparison Summary

This report summarizes the comparison between predicted spectrograms and groundtruth across all experiments.

## Experiments

EOF

# Collect metrics from all experiments
for metrics_file in spectrum_comparisons/*/metrics.txt; do
    if [ -f "$metrics_file" ]; then
        exp_name=$(dirname "$metrics_file" | xargs basename)
        
        echo "" >> "$summary_file"
        echo "### $exp_name" >> "$summary_file"
        echo "" >> "$summary_file"
        echo '```' >> "$summary_file"
        cat "$metrics_file" >> "$summary_file"
        echo '```' >> "$summary_file"
        echo "" >> "$summary_file"
        echo "![Visualization](${exp_name}/spectrum_comparison.png)" >> "$summary_file"
        echo "" >> "$summary_file"
    fi
done

echo ""
echo "=================================================="
echo "All comparisons complete!"
echo "Results saved to: spectrum_comparisons/"
echo "Summary report: $summary_file"
echo "=================================================="
