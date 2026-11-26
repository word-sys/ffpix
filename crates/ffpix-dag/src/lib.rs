//! Demand-driven pull-based execution engine for `ffpix`.

#![forbid(unsafe_code)]

pub mod graph;
pub mod op;

pub use graph::{DEFAULT_MAX_NODES, DagBuilder, DagGraph, DagNode};
pub use op::{
    AccessPattern, ErasedPullOp, ErasedState, EvalContext, NodeId, NoopPuller, OpKind, PullOp,
    PullOpAdapter, PullRequest, UpstreamPuller, erase_op,
};
