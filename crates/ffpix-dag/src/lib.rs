//! Demand-driven pull-based execution engine for `ffpix`.

#![forbid(unsafe_code)]

pub mod cache;
pub mod eval;
pub mod graph;
pub mod op;
pub mod strip_cache;
pub mod validate;

pub use cache::{TileCache, TileCacheKey, TileCacheStats};
pub use eval::PullSession;
pub use graph::{DEFAULT_MAX_NODES, DagBuilder, DagGraph, DagNode};
pub use op::{
    AccessPattern, ErasedPullOp, ErasedState, EvalContext, NodeId, NoopPuller, OpKind, PullOp,
    PullOpAdapter, PullRequest, UpstreamPuller, erase_op,
};
pub use strip_cache::{SharedStripCache, StripCache, StripCacheStats};
pub use validate::{ValidationReport, detect_cycles, validate_graph, validate_subgraph};
