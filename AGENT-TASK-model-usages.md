# Agent Task: Add model_usages Cost Tracking to DiscoveryState

## Goal

Add a `model_usages` reducer field to `DiscoveryState` and instrument each graph node that calls a model to emit usage records. This enables AstaBench cost tracking for the KRAKEN evaluation pipeline.

## Context

- KRAKEN's discovery pipeline is a LangGraph workflow in `backend/src/kestrel_backend/graph/`
- AstaBench requires accurate per-model cost tracking for Pareto frontier analysis (cost vs. performance)
- The pipeline already uses `Annotated[list[X], operator.add]` reducers extensively for parallel writes
- A separate repo (`kraken-chatbot-solver`) will consume this field to report costs to Inspect AI
- This is **the biggest engineering lift** for AstaBench integration (from the eval strategy doc)

## What to Build

### 1. ModelUsageRecord in state.py

Add a new Pydantic model and field to `DiscoveryState`:

```python
class ModelUsageRecord(BaseModel):
    """Record of a single model API call for cost tracking."""
    model_config = ConfigDict(frozen=True)
    
    model_name: str = Field(..., description="Model identifier (e.g., 'anthropic/claude-sonnet-4-6')")
    node_name: str = Field(..., description="Which graph node made this call")
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    cache_read_tokens: int = Field(0, ge=0, description="Prompt cache read tokens")
    cache_creation_tokens: int = Field(0, ge=0, description="Prompt cache creation tokens")
    
    def to_inspect_usage(self):
        """Convert to Inspect AI ModelUsage format."""
        from inspect_ai.model import ModelUsage
        return ModelUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
        )
```

Add to `DiscoveryState`:
```python
# === Cost Tracking ===
model_usages: Annotated[list[ModelUsageRecord], operator.add]
```

### 2. Instrument graph nodes

Find every node that calls a model (LLM API call). These are likely in `backend/src/kestrel_backend/graph/nodes/`. For each one:

1. **Identify the model call** — look for `anthropic` client calls, `langchain` LLM calls, or similar
2. **Capture the usage** — most LLM clients return usage info in the response (e.g., `response.usage.input_tokens`)
3. **Emit a ModelUsageRecord** — append to the state return dict

Example pattern:
```python
async def some_node(state: DiscoveryState) -> dict:
    # ... existing node logic ...
    response = await llm.ainvoke(prompt)
    
    # Add cost tracking
    usage_record = ModelUsageRecord(
        model_name="anthropic/claude-sonnet-4-6",  # or whatever model is configured
        node_name="some_node",
        input_tokens=response.usage_metadata.get("input_tokens", 0),
        output_tokens=response.usage_metadata.get("output_tokens", 0),
        cache_read_tokens=response.usage_metadata.get("cache_read_input_tokens", 0),
        cache_creation_tokens=response.usage_metadata.get("cache_creation_input_tokens", 0),
    )
    
    return {
        # ... existing return fields ...
        "model_usages": [usage_record],
    }
```

### 3. Investigate decorator/middleware approach

Before instrumenting each node individually, check if there's a cleaner way:
- Can LangGraph middleware/callbacks capture model usage automatically?
- Can a decorator wrap node functions to intercept LLM responses?
- The strategy doc raised this as open question Q6

If a decorator approach works, implement that instead of per-node changes. Document the decision either way.

## Key Files to Modify

- `backend/src/kestrel_backend/graph/state.py` — add ModelUsageRecord + field
- `backend/src/kestrel_backend/graph/nodes/*.py` — instrument each node (list all files in this directory first)

## Key Files to Read First

- `backend/src/kestrel_backend/graph/state.py` — understand existing state schema and patterns
- `backend/src/kestrel_backend/graph/builder.py` — understand graph structure
- `backend/src/kestrel_backend/graph/nodes/` — all node implementations, identify which ones call models
- `backend/src/kestrel_backend/graph/sdk_utils.py` — may contain shared LLM call utilities

## What NOT to do

- Don't change the graph structure or node logic — only add cost tracking instrumentation
- Don't add Inspect AI as a dependency to kraken-chatbot — the `to_inspect_usage()` method should use a lazy import
- Don't break existing tests — run `make check` or equivalent if it exists
- Don't change the existing node return signatures beyond adding `model_usages`

## Success Criteria

- `ModelUsageRecord` is defined in state.py
- `model_usages` field exists on DiscoveryState with operator.add reducer
- At least the nodes that make LLM calls emit usage records
- Existing tests still pass
- A note in the PR/commit about which nodes were instrumented and which (if any) don't make model calls
