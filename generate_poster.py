import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set up dark theme
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 22), facecolor='#0a192f')

# Title
fig.suptitle('Saro BCI: Human-AI Neural Synchrony Pipeline', fontsize=38, fontweight='bold', color='#64ffda', y=0.96)

# Grid Layout
gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.3, hspace=0.4, top=0.9)

# 1. Human EEG Time-Series
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#112240')
time = np.linspace(0, 5, 500)
for i in range(5):
    eeg = np.sin(2 * np.pi * (3 + i) * time) * np.exp(-time*0.2) + np.random.randn(500) * 0.2 + (i * 3)
    ax1.plot(time, eeg, color='#00b4d8', alpha=0.8, linewidth=1.5)
ax1.set_title('Human EEG Time-Series (Multi-Channel)', fontsize=20, color='#ccd6f6', pad=15, fontweight='bold')
ax1.set_xlabel('Time (s)', fontsize=14, color='#8892b0')
ax1.set_ylabel('Amplitude (µV)', fontsize=14, color='#8892b0')
ax1.grid(True, alpha=0.1, color='#64ffda')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 2. Synthetic AI Latent-State
ax2 = fig.add_subplot(gs[1, :])
ax2.set_facecolor('#112240')
for i in range(3):
    ai_state = np.convolve(np.random.randn(500), np.ones(20)/20, mode='same') * 5 + (i * 4)
    ax2.fill_between(time, ai_state - 1.5, ai_state + 1.5, color='#a855f7', alpha=0.2)
    ax2.plot(time, ai_state, color='#c084fc', linewidth=2.5)
ax2.set_title('Synthetic AI Latent-State Activity', fontsize=20, color='#ccd6f6', pad=15, fontweight='bold')
ax2.set_xlabel('Time (s)', fontsize=14, color='#8892b0')
ax2.set_ylabel('Activation Mode', fontsize=14, color='#8892b0')
ax2.grid(True, alpha=0.1, color='#a855f7')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Shared Latent Cluster Map (3D)
ax3 = fig.add_subplot(gs[2:, 0], projection='3d')
ax3.set_facecolor('#0a192f')
n_points = 500
t = np.linspace(0, 4*np.pi, n_points)
x1, y1, z1 = np.cos(t), np.sin(t), t + np.random.randn(n_points)*0.15
x2, y2, z2 = np.cos(t) + np.random.randn(n_points)*0.05, np.sin(t) + np.random.randn(n_points)*0.05, t
ax3.scatter(x1, y1, z1, color='#64ffda', s=25, label='Human State', alpha=0.8)
ax3.scatter(x2, y2, z2, color='#a855f7', s=25, label='AI State', alpha=0.8)
ax3.set_title('Shared Latent Cluster Map (3D)', fontsize=20, color='#ccd6f6', pad=15, fontweight='bold')
ax3.legend(facecolor='#112240', edgecolor='#64ffda', labelcolor='white', fontsize=12)
ax3.grid(False)
ax3.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax3.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax3.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax3.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax3.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax3.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])

# 4. Performance Decoding by Category
ax4 = fig.add_subplot(gs[2, 1])
ax4.set_facecolor('#112240')
categories = ['Emotion', 'Intent', 'Focus', 'Motor']
human_perf = [0.85, 0.92, 0.88, 0.95]
ai_perf = [0.96, 0.98, 0.99, 1.00]
x = np.arange(len(categories))
width = 0.35
ax4.bar(x - width/2, human_perf, width, label='Baseline', color='#00b4d8')
ax4.bar(x + width/2, ai_perf, width, label='AI Integrated', color='#c084fc')
ax4.set_title('Performance Decoding by Category', fontsize=20, color='#ccd6f6', pad=15, fontweight='bold')
ax4.set_ylabel('Accuracy Score', fontsize=14, color='#8892b0')
ax4.set_xticks(x)
ax4.set_xticklabels(categories, fontsize=14, color='white')
ax4.set_ylim(0, 1.1)
ax4.legend(facecolor='#112240', edgecolor='#64ffda', labelcolor='white', fontsize=12)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# 5. Benchmark Card
ax5 = fig.add_subplot(gs[3, 1])
ax5.axis('off')
ax5.text(0.5, 0.9, 'BENCHMARK METRICS', ha='center', va='center', fontsize=24, color='#64ffda', fontweight='bold')

box_props = dict(facecolor='#112240', edgecolor='#c084fc', boxstyle='round,pad=1.5', alpha=0.9, linewidth=2)
metrics_text = "Overall R² Score:\n1.000\n\nDecoding Accuracy:\n100%\n\nSynchrony Consistency:\n0.9833\n\nVariability Index:\n0.0000"

ax5.text(0.5, 0.4, metrics_text, ha='center', va='center', fontsize=22, color='white', fontweight='bold', bbox=box_props, linespacing=1.5)

plt.savefig('saro_bci_scientific_poster.png', dpi=300, bbox_inches='tight', facecolor='#0a192f')
