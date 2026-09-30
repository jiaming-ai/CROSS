# 📐 Reference View Selection Strategy

## 📖 Overview

Reference view selection is a component in multi-view depth estimation. When processing multiple input views, the model needs to determine which view should serve as the primary reference frame for depth prediction, defining the world coordinate system.

Different reference view will leads to different reconstruction results. This is a known consideration in multi-view geometry and was analyzed in [PI3](https://arxiv.org/abs/2507.13347). The choice of reference view can affect the quality and consistency of depth predictions across the scene.


## 🚀 Our Simple Solution: Automatic Reference View Selection

DA3 provides a simple approach to address this through **automatic reference view selection** based on **class tokens**. Instead of relying on heuristics or manual selection, the model analyzes the class token features from all input views and intelligently selects the most suitable reference frame.

---

## 🎨 Available Strategies

### 1. ⚖️ `saddle_balanced` (Recommended, Default)

**Philosophy:**  
Select a view that achieves balance across multiple feature metrics. This strategy looks for a "middle ground" view that is neither too similar nor too different from other views, making it a stable reference point.

**How it works:**
1. Extracts and normalizes class tokens from all views
2. Computes three complementary metrics for each view:
   - **Similarity score**: Average cosine similarity with other views
   - **Feature norm**: L2 norm of the original features  
   - **Feature variance**: Variance across feature dimensions
3. Normalizes each metric to [0, 1] range
4. Selects the view closest to 0.5 (median) across all three metrics

### 2. 🎢 `saddle_sim_range`

**Philosophy:**  
Select a view with the largest similarity range to other views. This identifies "saddle point" views that are highly similar to some views but dissimilar to others, making them information-rich anchor points.

**How it works:**
1. Computes pairwise cosine similarity between all views
2. For each view, calculates the range (max - min) of similarities to other views
3. Selects the view with the maximum similarity range

---

### 3. 1️⃣ `first` (Not Recommended)

**Philosophy:**  
Always use the first view in the input sequence as the reference.

**How it works:**
Simply returns index 0.

**When to use:**
- ⛔ **Not recommended** in general
- 🔧 Only use when you have manually pre-sorted your views and know the first view is optimal
- 🐛 Debugging or baseline comparisons

---

### 4. ⏸️ `middle`

**Philosophy:**  
Select the view in the middle of the input sequence.

**How it works:**
Returns the view at index `S // 2` where S is the number of views.

**When to use:**
- ⏱️ **Only recommended when input images are temporally ordered**
- 🎬 Video sequences (e.g., **DA3-LONG** setting)
- 📹 Sequential captures where the middle frame likely has the most stable viewpoint

**Specific use case: DA3-LONG** 🎬  
In video-based depth estimation scenarios (like DA3-LONG), where inputs are consecutive frames, `middle` is often the **optimal choice** because that it has maximum overlap with all other frames.


## 💻 Usage

### 🐍 Python API

```python
from depth_anything_3 import DepthAnything3

model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")

# Use default (saddle_balanced)
prediction = model.inference(
    images,
    ref_view_strategy="saddle_balanced"
)

# For video sequences, consider using middle
prediction = model.inference(
    video_frames,
    ref_view_strategy="middle"  # Good for temporal sequences
)

# For complex scenes with wide baselines
prediction = model.inference(
    images,
    ref_view_strategy="saddle_sim_range"
)
```

### 🖥️ Command Line Interface

```bash
# Default (saddle_balanced)
da3 auto input/ --export-dir output/

# Explicitly specify strategy
da3 auto input/ --ref-view-strategy saddle_balanced

# For video processing
da3 video input.mp4 --ref-view-strategy middle

# For wide-baseline multi-view
da3 images captures/ --ref-view-strategy saddle_sim_range
```

---

### 🎯 When Selection Is Applied

Reference view selection is applied when:
- 3️⃣ Number of views S ≥ 3

---

## 💡 Recommendations

### 📋 Quick Guide

| Scenario | Recommended Strategy | Rationale |
|----------|---------------------|-----------|
| **Default / Unknown** | `saddle_balanced` | Robust, balanced, works well across diverse scenarios |
| **Video frames** | `middle` | Temporal coherence, stable middle frame |
| **Wide-baseline multi-view** | `saddle_sim_range` | Maximizes information coverage |
| **Pre-sorted inputs** | `first` | Use only if you've manually optimized ordering |
| **Single image** | `first` | Automatically used (no reordering needed for S ≤ 2) |

### ✨ Best Practices

1. 🎯 **Start with defaults**: `saddle_balanced` works well in most cases
2. 🎬 **Consider your input type**: Use `middle` for videos, `saddle_balanced` for photos
3. 🔬 **Experiment if needed**: Try different strategies if results are suboptimal
4. 📊 **Monitor performance**: Check `glb` quality and consistency across views.

---

## 🔧 Technical Details

### 🎚️ Selection Threshold

The reference view selection is only triggered when:
```python
num_views >= 3  # At least 3 views required
```

For 1-2 views, no reordering is performed (equivalent to using `first`).

### ⚙️ Implementation

The selection happens at layer `alt_start - 1` in the vision transformer, before the first global attention layer. This ensures the selected reference view influences the entire depth prediction pipeline.

---

## ❓ FAQ

**Q: 🤔 Why is this feature provided?**  
A: The model can handle any view order, but this feature provides automatic optimization for reference view selection, which can help improve depth prediction quality in multi-view scenarios.

**Q: ⏱️ Does this add computational cost?**  
A: The overhead is totally negligible.

**Q: 🎮 Can I manually specify which view to use as reference?**  
A: Not directly through this parameter. You can pre-sort your input images to place your preferred reference view first and use `ref_view_strategy="first"`.

**Q: ⚙️ What happens if I don't specify this parameter?**  
A: The default `saddle_balanced` strategy is used automatically.

**Q: 📊 Is this feature used in the DA3 paper benchmarks?**  
A: No, the paper used `first` as the default strategy for all multi-view experiments. The current default has been updated to `saddle_balanced` for better robustness.

