# Cold-Start Performance Optimization Summary

## Issue #10: Cold-Start SDK Serialization Optimization

### Changes Implemented

#### 1. Constant Adjustments (`cold_start.py`)

| Constant | Before | After | Rationale |
|----------|--------|-------|-----------|
| `BATCH_SIZE` | 3 | 5 | Increased parallelism for faster processing |
| `SDK_SEMAPHORE` | 6 | 8 | Allow more concurrent SDK calls |
| `SDK_INFERENCE_TIMEOUT` | 120s | 60s | Reduce wait time for timeouts |
| `ANALOGUE_LIMIT` | 5 | 3 | Fewer analogues = smaller prompts = faster inference |

#### 2. Entity Limiting Strategy

**New behavior**: Process only top 5 sparse + top 3 cold-start entities by edge count

- **Added**: `score_entity_complexity()` function to prioritize entities
  - Cold-start entities (0 edges) get highest priority (score 0.0)
  - Sparse entities sorted by edge count (lower = higher priority)

- **Modified**: `run()` function to:
  - Score and sort all entities before processing
  - Limit to MAX_SPARSE=5 and MAX_COLD_START=3
  - Log skipped count for visibility
  - Return `cold_start_skipped_count` in state

**Trade-off**: May miss important sparse entities with higher edge counts. However, entities with more edges are better handled by the direct_kg node, so this focuses cold-start analysis where it provides the most value.

#### 3. Early Termination for Low-Quality Analogues

**New behavior**: Skip SDK inference if no analogues have similarity >= 0.7

- **Added**: Quality check in `analyze_cold_start_entity()`
  - Filters analogues to those with similarity >= 0.7
  - If no quality analogues, returns basic Finding without SDK inference
  - Still returns analogue information for visibility

**Trade-off**: May skip potentially useful inferences from moderate-similarity analogues (0.5-0.7 range). However, low-similarity analogues often lead to noisy or incorrect inferences, and this saves significant computation time.

#### 4. Connection Pooling (`kestrel_client.py`)

**New behavior**: HTTP client with connection pooling and HTTP/2 support

- **Added**: `httpx.Limits` configuration
  - `max_keepalive_connections=20`: Reuse up to 20 idle connections
  - `max_connections=100`: Allow up to 100 total connections
  - `keepalive_expiry=30.0`: Keep connections alive for 30 seconds

- **Added**: Conditional HTTP/2 support
  - Checks for h2 package availability
  - Falls back gracefully if not installed
  - Enables multiplexing when available

**Trade-off**: Minimal - connection pooling is nearly always beneficial. HTTP/2 requires the h2 package, but we gracefully degrade to HTTP/1.1 if unavailable.

#### 5. State Schema Addition (`state.py`)

**Added**: `cold_start_skipped_count: int` field to DiscoveryState

- Tracks number of entities skipped by optimization
- Provides visibility into what's being filtered
- Can inform future tuning decisions

### Performance Impact

**Expected improvements**:
- 40-60% reduction in cold-start processing time
- Reduced memory footprint from smaller prompts
- Better resource utilization through connection pooling
- Faster failure detection with reduced timeout

**Test coverage**:
- `test_cold_start_performance.py`: 9 tests covering all new functionality
  - Entity complexity scoring
  - Entity limiting (top 5 sparse + top 3 cold-start)
  - Prioritization by edge count
  - Early termination for low-quality analogues
  - Integration test: 10 sparse entities under 5 minutes

### Trade-offs and Monitoring

#### Key Trade-offs

1. **Entity Limiting**: Focusing on top 5+3 entities may miss important sparse entities
   - **Mitigation**: Entities with more edges are better handled by direct_kg node
   - **Monitor**: `cold_start_skipped_count` in logs

2. **Reduced Timeout**: 60s timeout may cause more failures than 120s
   - **Mitigation**: Increased parallelism and smaller prompts should reduce processing time
   - **Monitor**: Timeout error rates in logs

3. **Quality Threshold**: Similarity >= 0.7 cutoff may skip moderate-quality inferences
   - **Mitigation**: Low-similarity analogues often produce noisy results anyway
   - **Monitor**: "skipped inference" findings in results

#### Recommended Monitoring

- Watch `cold_start_skipped_count` to see how many entities are being filtered
- Track timeout rates to ensure 60s is sufficient
- Monitor "max similarity < 0.7" findings to assess quality threshold impact
- Compare cold-start duration before/after to quantify improvements

### Files Modified

1. `/backend/src/kestrel_backend/graph/nodes/cold_start.py`
   - Adjusted constants
   - Added `score_entity_complexity()` function
   - Added early termination logic
   - Modified `run()` to limit and prioritize entities

2. `/backend/src/kestrel_backend/kestrel_client.py`
   - Added connection pooling configuration
   - Added conditional HTTP/2 support

3. `/backend/src/kestrel_backend/graph/state.py`
   - Added `cold_start_skipped_count` field

### Files Added

1. `/backend/tests/test_cold_start_performance.py`
   - Comprehensive test suite for performance optimizations
   - 9 tests covering all new functionality
   - Integration test for 10-entity performance benchmark

### Next Steps

1. Monitor production performance after deployment
2. Tune constants if needed based on observed behavior
3. Consider making limits configurable via environment variables
4. Potentially add metrics tracking for cold-start performance
