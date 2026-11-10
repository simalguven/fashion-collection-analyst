# Agent evaluation: final

Passed 39 of 40 runs (98%). Tokens used: 88,804. Median answer time: 12.05s.

| Category | Passed |
|---|---|
| happy_path | 14/14 |
| multi_step | 6/6 |
| edge_case | 9/10 |
| out_of_scope | 6/6 |
| adversarial | 4/4 |

## Failed runs

**unknown_colour** (edge_case)
- Question: Which cluster has the most ultraviolet items?
- Expected: says the colour is not tracked
- Tools called: find_clusters_by_colour, find_clusters_by_colour
- Tools ok: True, answer ok: False
- Answer: There's no "ultraviolet" colour in this collection — the closest tracked colour is **Purple**. Ranked by the share of items whose main colour is Purple: 1. **Cluster 6 – Watches** (2% of its items) 2. **Cluster 9 – Tshirts (Boys)** (0.9%) 3. Clusters 0 (Shirts), 1 (Perfume & Body Mist/Deodorant), and 2 (Heels & Flip Flops) — 0% So **Cluster 6 (Watches)** has the most purple items by share. If you 
