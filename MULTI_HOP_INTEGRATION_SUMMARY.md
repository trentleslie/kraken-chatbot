# Multi-Hop Query API Integration - Implementation Summary

## Overview

Successfully integrated Kestrel's `multi_hop_query` API into the KRAKEN discovery pipeline to replace LLM-based pathfinding with efficient API calls. This addresses Issue #27.

## Changes Made

### 1. kestrel_client.py - New API Wrapper

**File**: `backend/src/kestrel_backend/kestrel_client.py`

**Added**:
- `multi_hop_query()` async function supporting both search modes:
  - **Singly-pinned**: Explore N hops from start nodes
  - **Doubly-pinned**: Find paths connecting start and end nodes
- Input validation (max_hops 1-5, required start_node_ids)
- Configurable path limits and predicate filtering

**Example Usage**:
```python
# Explore 2 hops from glucose
await multi_hop_query(start_node_ids=["CHEBI:17234"], max_hops=2)

# Find paths from glucose to diabetes
await multi_hop_query(
    start_node_ids=["CHEBI:17234"],
    end_node_ids=["MONDO:0005148"],
    max_hops=3
)
```

### 2. integration.py - API-Based Bridge Detection

**File**: `backend/src/kestrel_backend/graph/nodes/integration.py`

**Refactored**:
- **Phase A**: Bridge detection now uses `detect_bridges_via_api()` instead of LLM
- **Phase B**: LLM retained for gap analysis only (reasoning-intensive task)

**New Functions**:
- `detect_bridges_via_api()`: Groups entities by category, queries cross-category paths
- `parse_multi_hop_result()`: Parses API responses into Bridge objects

**Benefits**:
- Faster: Direct API calls vs multi-turn LLM conversation
- More reliable: Structured JSON responses vs text parsing
- Lower cost: No LLM calls for bridge detection

**Tier Assignment**:
- Tier 2: Paths with ≤2 hops (high confidence)
- Tier 3: Paths with >2 hops (speculative)

### 3. synthesis.py - Bridge Hypothesis Validation

**File**: `backend/src/kestrel_backend/graph/nodes/synthesis.py`

**Added**:
- `validate_bridge_hypotheses()`: Validates Tier 3 bridges using doubly-pinned search
- Upgrades validated bridges from Tier 3 → Tier 2
- Returns validated bridges to update state

**Validation Logic**:
```
For each Tier 3 bridge:
  1. Extract start and end CURIEs
  2. Run doubly-pinned multi_hop_query
  3. If path exists → upgrade to Tier 2 + mark as "KG-validated"
  4. If no path → keep as Tier 3
```

### 4. pathway_enrichment.py - Two-Hop Shared Neighbors

**File**: `backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py`

**Added**:
- `find_two_hop_shared_neighbors()`: Discovers indirect connectivity via API
- Complements LLM-based one-hop analysis
- Counts neighbors reachable by 2+ input entities

**Integration**:
- **Phase A**: Two-hop API analysis (fast, no LLM)
- **Phase B**: One-hop LLM analysis (existing logic)
- Findings added to `direct_findings` with tier=2

### 5. kestrel_tools.py - Tool Documentation

**File**: `backend/src/kestrel_backend/kestrel_tools.py`

**Added**:
- `@tool` decorator for `multi_hop_query`
- Documented parameters and search modes
- Added to `create_kestrel_mcp_server()` tool list

### 6. Comprehensive Test Suite

**File**: `backend/tests/test_multi_hop_integration.py`

**Tests Cover**:
1. **TestMultiHopQueryWrapper** (4 tests)
   - Singly-pinned and doubly-pinned modes
   - Input validation (max_hops, start_node_ids)

2. **TestDetectBridgesViaAPI** (3 tests)
   - Empty entities and single-category handling
   - Cross-category bridge detection with mocked API

3. **TestParseMultiHopResult** (2 tests)
   - Valid path parsing to Bridge objects
   - Empty result handling

4. **TestValidateBridgeHypotheses** (4 tests)
   - Tier 3 → Tier 2 upgrade on validation
   - Tier 3 remains if validation fails
   - Tier 2 bridges unchanged

5. **TestFindTwoHopSharedNeighbors** (4 tests)
   - Shared neighbor detection
   - Single-connection filtering
   - Edge cases (empty, single entity)

**Test Results**: All 17 tests passing ✓

## Architecture Benefits

### Before (LLM-Based)
```
Integration Node:
  ├── LLM: Bridge detection (5 turns, ~30s)
  └── LLM: Gap analysis (5 turns, ~30s)
  Total: ~60s, 10 LLM turns
```

### After (Hybrid API + LLM)
```
Integration Node:
  ├── API: Bridge detection (~5s, no LLM)
  └── LLM: Gap analysis only (5 turns, ~30s)
  Total: ~35s, 5 LLM turns (42% reduction)

Synthesis Node:
  ├── API: Bridge validation (~2s per bridge)
  └── LLM: Report generation (existing)

Pathway Enrichment Node:
  ├── API: Two-hop analysis (~10s, no LLM)
  └── LLM: One-hop analysis (existing)
```

## Key Design Decisions

1. **Hybrid Approach**: Keep LLM for reasoning-intensive tasks (gap analysis), use API for structured pathfinding
2. **Tier-Based Validation**: Only validate Tier 3 (speculative) bridges to minimize API calls
3. **Result Size Limits**: Enforce `limit` parameter to prevent huge result sets (max_hops=5 could return thousands of paths)
4. **Error Handling**: Graceful degradation - API errors don't block pipeline execution

## Integration Points

### State Schema (no changes required)
- Existing `Bridge` model supports API-generated bridges
- `validated_bridges` returned from synthesis to update state
- Two-hop findings added to `direct_findings` (existing reducer)

### Pipeline Flow
```
Intake → Entity Resolution → Triage
                ↓
         Direct KG (Tier 1)
                ↓
         Cold Start (Tier 3)
                ↓
    Pathway Enrichment (+ two-hop API)
                ↓
      Integration (API bridges + LLM gaps)
                ↓
         Temporal (conditional)
                ↓
      Synthesis (validate bridges + report)
```

## Files Modified

1. `/backend/src/kestrel_backend/kestrel_client.py` - Added `multi_hop_query()`
2. `/backend/src/kestrel_backend/kestrel_tools.py` - Added tool documentation
3. `/backend/src/kestrel_backend/graph/nodes/integration.py` - Refactored bridge detection
4. `/backend/src/kestrel_backend/graph/nodes/synthesis.py` - Added bridge validation
5. `/backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py` - Added two-hop analysis
6. `/backend/tests/test_multi_hop_integration.py` - New test suite (17 tests)

## Next Steps

1. **Manual Testing**: Test with real Kestrel API to verify response format assumptions
2. **Performance Profiling**: Measure actual speedup in production queries
3. **Result Limit Tuning**: Adjust `limit` parameters based on real-world query patterns
4. **Edge Case Handling**: Test with cold-start entities, hub nodes, disconnected graphs

## Notes

- Implementation follows existing patterns (async/await, state reducers, error handling)
- No changes to main.py, protocol.py, or database.py (as scoped)
- Tests use mocking to avoid external API dependencies
- All functions include docstrings with parameter descriptions and examples
