"""
Generate core figures for the second paper (G1-G4 results)

Priority order:
1. Fig.1: Overall STCV-Occ Framework
2. Fig.4: G3 Cooperation Utility Predictor Architecture
3. Fig.5: G4 Learned Fusion Mechanism (Bar chart + BA/HA comparison)
4. Fig.2: G1 Negative Cooperation Diagnosis
5. Fig.3: G2 Oracle Headroom Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import json
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Output directory
FIG_DIR = Path('/root/autodl-tmp/UniV2X/G0-G1-G2/G4/figures')
FIG_DIR.mkdir(exist_ok=True)

# Load G4 results
with open('/root/autodl-tmp/UniV2X/G0-G1-G2/G4/results/g4_final_results.json') as f:
    g4_results = json.load(f)

with open('/root/autodl-tmp/UniV2X/G0-G1-G2/G4/results/g4_mechanism_analysis.json') as f:
    g4_mechanism = json.load(f)

print("="*80)
print("Generating Figures for Second Paper")
print("="*80)

# ============================================================================
# Fig.5: G4 Learned Fusion Results (MOST IMPORTANT - Do first!)
# ============================================================================

def plot_fig5_g4_results():
    """
    Fig.5: G4 Learned Fusion Mechanism
    - Bar chart: IoU comparison (Official, Ego, Learned, Oracle)
    - Side panel: BA Retention and HA Admission comparison
    """
    fig = plt.figure(figsize=(12, 5))

    # Left panel: IoU comparison
    ax1 = plt.subplot(1, 2, 1)

    test_results = g4_results['test_strategy_results']
    strategies = ['Official OR', 'Ego', 'Learned', 'Oracle']
    ious = [
        test_results['official']['iou'],
        test_results['ego']['iou'],
        test_results['learned']['iou'],
        test_results['oracle']['iou']
    ]

    colors = ['#94a3b8', '#64748b', '#3b82f6', '#10b981']  # Gray, darker gray, blue, green

    bars = ax1.bar(strategies, ious, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, iou) in enumerate(zip(bars, ious)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{iou:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight the key ordering with arrows
    ax1.annotate('', xy=(2, ious[2]), xytext=(0, ious[0]),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--'))
    ax1.text(1, (ious[0] + ious[2])/2 + 0.01, '+8.2%',
            ha='center', fontsize=9, color='red', fontweight='bold')

    ax1.set_ylabel('IoU (Intersection over Union)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Downstream Fusion Performance', fontsize=12, fontweight='bold')
    ax1.set_ylim([0.24, 0.32])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add horizontal line at Official level
    ax1.axhline(y=ious[0], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    # Right panel: BA Retention and HA Admission
    ax2 = plt.subplot(1, 2, 2)

    learned_stats = g4_mechanism['learned_predictor']['test']['overall']
    conf_stats = g4_mechanism['confidence_baseline']['test']['0.3']  # δ=0.30

    x = np.arange(2)
    width = 0.35

    learned_vals = [learned_stats['ba_retention'] * 100, learned_stats['ha_admission'] * 100]
    conf_vals = [conf_stats['ba_retention'] * 100, conf_stats['ha_admission'] * 100]

    bars1 = ax2.bar(x - width/2, learned_vals, width, label='Learned (τ=0.70)',
                   color='#3b82f6', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, conf_vals, width, label='Confidence (δ=0.30)',
                   color='#f97316', alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Highlight the difference in HA Admission
    ax2.annotate('', xy=(1 - width/2, learned_vals[1]), xytext=(1 + width/2, conf_vals[1]),
                arrowprops=dict(arrowstyle='<->', lw=2, color='green'))
    ax2.text(1, (learned_vals[1] + conf_vals[1])/2, '-6.88%\n(30% ↓)',
            ha='center', fontsize=8, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='green'))

    ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Decision Quality Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['BA Retention\n(Recall)', 'HA Admission\n(False Positive Rate)'])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim([0, 45])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    plt.tight_layout()

    # Save
    for fmt in ['pdf', 'png']:
        fig.savefig(FIG_DIR / f'fig5_g4_learned_fusion.{fmt}', dpi=300, bbox_inches='tight')

    print("\n✓ Fig.5: G4 Learned Fusion Results")
    print(f"  Saved to: {FIG_DIR}/fig5_g4_learned_fusion.{{pdf,png}}")
    print(f"  Key message: Learned > Ego > Official, with 30% lower HA admission vs baseline")

    plt.close()

# ============================================================================
# Fig.4: G3 Cooperation Utility Predictor Architecture
# ============================================================================

def plot_fig4_predictor_architecture():
    """
    Fig.4: G3 Cooperation Utility Predictor Architecture
    Compact and clear diagram showing 4 feature groups → MLP
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Title
    ax.text(5, 9.5, 'Cooperation Utility Predictor Architecture',
           ha='center', fontsize=14, fontweight='bold')

    # Feature groups (left side)
    feature_boxes = [
        {'name': 'Local Soft\nEvidence', 'items': ['Pv', 'Pi', 'ΔP', '|ΔP|', 'Warp'],
         'dim': '80-dim', 'y': 7.5, 'color': '#3b82f6'},
        {'name': 'Horizon\nEncoding', 'items': ['t=0,1,2,3,4'],
         'dim': '5-dim', 'y': 5.8, 'color': '#8b5cf6'},
        {'name': 'Spatial\nPosition', 'items': ['u, v (normalized)'],
         'dim': '2-dim', 'y': 4.1, 'color': '#f59e0b'},
        {'name': 'Neighborhood\nContext', 'items': ['8 neighbors × 5 stats'],
         'dim': '40-dim', 'y': 2.4, 'color': '#10b981'}
    ]

    for box in feature_boxes:
        # Main box
        rect = FancyBboxPatch((0.5, box['y']-0.6), 2.5, 1.2,
                             boxstyle="round,pad=0.05",
                             facecolor=box['color'], alpha=0.15,
                             edgecolor=box['color'], linewidth=2)
        ax.add_patch(rect)

        # Name
        ax.text(1.75, box['y']+0.3, box['name'],
               ha='center', va='center', fontsize=10, fontweight='bold')

        # Items
        items_text = '\n'.join(box['items'])
        ax.text(1.75, box['y']-0.2, items_text,
               ha='center', va='center', fontsize=8)

        # Dimension label
        ax.text(3.2, box['y'], box['dim'],
               ha='left', va='center', fontsize=8, style='italic',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Arrow to concatenation
        arrow = FancyArrowPatch((3.0, box['y']), (4.2, 5.0),
                               arrowstyle='->', lw=1.5, color='gray',
                               connectionstyle="arc3,rad=0.1")
        ax.add_patch(arrow)

    # Concatenation node
    concat_circle = Circle((4.5, 5.0), 0.3, facecolor='#64748b', alpha=0.3,
                          edgecolor='#64748b', linewidth=2)
    ax.add_patch(concat_circle)
    ax.text(4.5, 5.0, '⊕', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(4.5, 4.3, '127-dim', ha='center', va='top', fontsize=8, style='italic')

    # MLP layers (right side)
    mlp_layers = [
        {'dim': 256, 'y': 7.0},
        {'dim': 128, 'y': 5.5},
        {'dim': 64, 'y': 4.0},
        {'dim': 1, 'y': 2.5}
    ]

    # Arrow from concat to first layer
    arrow = FancyArrowPatch((4.8, 5.0), (5.7, mlp_layers[0]['y']),
                           arrowstyle='->', lw=2, color='#0f172a')
    ax.add_patch(arrow)

    prev_y = mlp_layers[0]['y']
    for i, layer in enumerate(mlp_layers):
        # Layer box
        width = 1.5 if layer['dim'] > 1 else 1.0
        rect = Rectangle((6.5-width/2, layer['y']-0.3), width, 0.6,
                        facecolor='#0f172a', alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)

        # Dimension text
        ax.text(6.5, layer['y'], f"{layer['dim']}",
               ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Activation
        if i < len(mlp_layers) - 1:
            ax.text(7.8, layer['y'], 'ReLU + Dropout(0.1)',
                   ha='left', va='center', fontsize=7, style='italic')
        else:
            ax.text(7.8, layer['y'], 'Sigmoid',
                   ha='left', va='center', fontsize=7, style='italic', color='#10b981')

        # Arrow to next layer
        if i < len(mlp_layers) - 1:
            arrow = FancyArrowPatch((6.5, layer['y']-0.3), (6.5, mlp_layers[i+1]['y']+0.3),
                                   arrowstyle='->', lw=2, color='#0f172a')
            ax.add_patch(arrow)

    # Output label
    ax.text(6.5, 1.5, 'sR ∈ [0,1]', ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#10b981', alpha=0.3, edgecolor='#10b981', linewidth=2))
    ax.text(6.5, 1.0, '(Cooperation Utility Score)', ha='center', fontsize=8, style='italic')

    # Total parameters
    ax.text(5, 0.3, 'Total Parameters: 73,985',
           ha='center', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='#f1f5f9', edgecolor='gray', linewidth=1))

    plt.tight_layout()

    # Save
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(FIG_DIR / f'fig4_g3_predictor_architecture.{fmt}', dpi=300, bbox_inches='tight')

    print("\n✓ Fig.4: G3 Predictor Architecture")
    print(f"  Saved to: {FIG_DIR}/fig4_g3_predictor_architecture.{{pdf,png,svg}}")
    print(f"  Key message: 4 feature groups (127-dim) → Lightweight MLP → Utility score")

    plt.close()

# ============================================================================
# Generate all figures
# ============================================================================

if __name__ == '__main__':
    print("\nGenerating figures in priority order...\n")

    # Priority 1: G4 results (most important for convincing readers)
    plot_fig5_g4_results()

    # Priority 2: G3 predictor architecture (core method)
    plot_fig4_predictor_architecture()

    print("\n" + "="*80)
    print("✓ Phase 1 figures complete (Fig.4, Fig.5)")
    print("="*80)
    print("\nNext steps:")
    print("  • Fig.1: Overall framework diagram (requires manual design)")
    print("  • Fig.2: G1 negative cooperation diagnosis")
    print("  • Fig.3: G2 Oracle headroom analysis")
    print("  • Qualitative visualizations (after G5 integration)")
    print("")
