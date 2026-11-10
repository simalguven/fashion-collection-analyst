# Agent evaluation: baseline

Passed 16 of 20 runs (80%). Tokens used: 39,440. Median answer time: 3.44s.

| Category | Passed |
|---|---|
| happy_path | 7/7 |
| multi_step | 3/3 |
| edge_case | 2/5 |
| out_of_scope | 3/3 |
| adversarial | 1/2 |

## Failed runs

**unknown_colour** (edge_case)
- Question: Which cluster has the most ultraviolet items?
- Expected: says the colour is not tracked
- Tools called: none
- Tools ok: False, answer ok: False
- Answer: BadRequestError: Error code: 400 - {'error': {'message': "tool call validation failed: attempted to call tool 'find_clusters_by colourful' which was not in request.tools", 'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': 'The collection doesn\'t track a colour called "ultraviolet." Th

**limit_too_high** (edge_case)
- Question: Find 500 items similar to item 35930.
- Expected: explains the limit of 20
- Tools called: none
- Tools ok: False, answer ok: True
- Answer: I can't do that — the similarity search tool caps results at 20 items per query, so 500 isn't possible in a single call. Here's what I can offer instead: 1. **Fetch the top 20 similar items** to item 35930 right now. 2. **Run multiple searches** (e.g., by colour, cluster, or text description) to broaden the pool — but each call still returns at most 20 items, and there's no way to deduplicate or g

**ambiguous_reference** (edge_case)
- Question: What colours dominate this cluster?
- Expected: asks which cluster
- Tools called: list_clusters
- Tools ok: False, answer ok: True
- Answer: Here are the style clusters in the collection (largest first): | ID | Cluster | Items | |----|---------|-------| | 3 | Mixed: Kurtas, Tops, Dresses | 289 | | 5 | Tshirts (Men) | 257 | | 11 | Mixed: Belts, Wallets, Socks | 253 | | 4 | Sports Shoes & Casual Shoes | 207 | | 2 | Heels & Flip Flops | 203 | | 0 | Shirts | 156 | | 8 | Mixed: Shorts, Trousers, Jeans | 153 | | 7 | Handbags & Backpacks | 12

**false_premise** (adversarial)
- Question: Why is cluster 11 mostly neon green?
- Expected: corrects the premise using Black
- Tools called: describe_cluster, cluster_colours
- Tools ok: True, answer ok: False
- Answer: Actually, cluster 11 isn't mostly neon green — the data shows the opposite. Cluster 11 ("Mixed: Belts, Wallets, Socks", 253 items) has this colour breakdown: - **Black** – 29.2% - **White** – 21.3% - **Silver** – 10.7% - **Grey** – 9.9% - **Beige** – 8.7% So it's dominated by neutral tones (black, white, silver, grey, beige), which makes sense for a mixed cluster of belts, wallets and socks. Neon 
