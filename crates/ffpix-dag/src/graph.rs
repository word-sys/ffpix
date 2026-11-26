//! Immutable `DagGraph`, `DagNode`, and checked `DagBuilder` with deterministic hashing.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use ffpix_core::{Budget, BudgetKind, BudgetLimits, FfpError, FfpResult, ImageDescriptor, Sha256};

use crate::op::{AccessPattern, ErasedPullOp, NodeId, OpKind, PullOp, erase_op};

pub const DEFAULT_MAX_NODES: usize = 4096;

#[derive(Debug, Clone)]
pub struct DagNode {
    pub id: NodeId,
    pub op: Arc<dyn ErasedPullOp>,
    pub inputs: Vec<NodeId>,
}

impl DagNode {
    #[must_use]
    pub const fn id(&self) -> NodeId {
        self.id
    }

    #[must_use]
    pub fn op(&self) -> &dyn ErasedPullOp {
        self.op.as_ref()
    }

    #[must_use]
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    #[must_use]
    pub fn descriptor(&self) -> &ImageDescriptor {
        self.op.descriptor()
    }

    #[must_use]
    pub fn access_pattern(&self) -> AccessPattern {
        self.op.access_pattern()
    }

    #[must_use]
    pub fn op_kind(&self) -> OpKind {
        self.op.op_kind()
    }

    #[must_use]
    pub fn name(&self) -> &'static str {
        self.op.name()
    }
}

#[derive(Debug, Clone)]
pub struct DagGraph {
    nodes: Vec<DagNode>,
    outputs: Vec<(String, NodeId)>,
}

impl DagGraph {
    /// Constructs a graph directly from raw nodes and outputs without validating edges.
    /// Intended for optimizer passes and validator cycle tests.
    #[must_use]
    pub fn from_parts_unvalidated(nodes: Vec<DagNode>, outputs: Vec<(String, NodeId)>) -> Self {
        Self { nodes, outputs }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: NodeId) -> FfpResult<&DagNode> {
        self.nodes
            .get(id.index())
            .ok_or_else(|| FfpError::invalid_argument(format!("unknown {id}")))
    }

    #[must_use]
    pub fn nodes(&self) -> &[DagNode] {
        &self.nodes
    }

    #[must_use]
    pub fn outputs(&self) -> &[(String, NodeId)] {
        &self.outputs
    }

    #[must_use]
    pub fn output(&self, name: &str) -> Option<NodeId> {
        self.outputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, id)| *id)
    }

    #[must_use]
    pub fn default_output(&self) -> Option<NodeId> {
        if let Some(id) = self.output("output") {
            return Some(id);
        }
        if let Some((_, id)) = self.outputs.last() {
            return Some(*id);
        }
        self.nodes.last().map(|n| n.id)
    }

    #[must_use]
    pub fn consumers(&self, target: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        for node in &self.nodes {
            if node.inputs.contains(&target) {
                out.push(node.id);
            }
        }
        out
    }

    /// Combines access patterns across `root` and all its upstream ancestors iteratively.
    pub fn effective_access_pattern(&self, root: NodeId) -> FfpResult<AccessPattern> {
        let _ = self.node(root)?;
        let mut visited = vec![false; self.nodes.len()];
        let mut stack = vec![root];
        let mut pattern = AccessPattern::Random;

        while let Some(id) = stack.pop() {
            let idx = id.index();
            if visited.get(idx).copied().unwrap_or(true) {
                continue;
            }
            if let Some(slot) = visited.get_mut(idx) {
                *slot = true;
            }
            let node = self.node(id)?;
            pattern = pattern.combine(node.access_pattern());
            for &inp in &node.inputs {
                let _ = self.node(inp)?;
                stack.push(inp);
            }
        }
        Ok(pattern)
    }

    /// Returns a deterministic topological ordering (sources before consumers),
    /// breaking ties by ascending `NodeId`. Fails with `CycleDetected` on cyclic graphs.
    pub fn topological_order(&self) -> FfpResult<Vec<NodeId>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (idx, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs {
                let src_idx = inp.index();
                if src_idx >= n {
                    return Err(FfpError::invalid_argument(format!(
                        "{} references unknown input {inp}",
                        node.id
                    )));
                }
                in_degree[idx] += 1;
                dependents[src_idx].push(idx);
            }
        }

        let mut ready = BinaryHeap::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                ready.push(Reverse(idx));
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(Reverse(idx)) = ready.pop() {
            order.push(self.nodes[idx].id);
            for &dep in &dependents[idx] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    ready.push(Reverse(dep));
                }
            }
        }

        if order.len() != n {
            return Err(FfpError::cycle_detected(
                "cycle detected while computing topological order",
            ));
        }
        Ok(order)
    }

    /// Computes a deterministic structural SHA-256 hash over nodes, edges,
    /// descriptors, operator fingerprints, and named outputs.
    #[must_use]
    pub fn graph_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"ffpix-dag-v1");
        h.update(&(self.nodes.len() as u64).to_le_bytes());

        for node in &self.nodes {
            h.update(&node.id.0.to_le_bytes());
            let name = node.name().as_bytes();
            h.update(&(name.len() as u32).to_le_bytes());
            h.update(name);
            h.update(&[node.op_kind() as u8, node.access_pattern() as u8]);

            let desc = node.descriptor();
            h.update(&desc.width.to_le_bytes());
            h.update(&desc.height.to_le_bytes());
            h.update(&[
                desc.pixel_format.layout as u8,
                desc.pixel_format.sample as u8,
                desc.color_space as u8,
                desc.alpha_mode as u8,
                desc.orientation as u8,
            ]);
            h.update(&desc.frame_count.to_le_bytes());

            h.update(&(node.inputs.len() as u32).to_le_bytes());
            for inp in &node.inputs {
                h.update(&inp.0.to_le_bytes());
            }
            node.op.fingerprint(&mut h);
        }

        h.update(&(self.outputs.len() as u64).to_le_bytes());
        for (name, id) in &self.outputs {
            let b = name.as_bytes();
            h.update(&(b.len() as u32).to_le_bytes());
            h.update(b);
            h.update(&id.0.to_le_bytes());
        }

        h.finalize()
    }
}

#[derive(Debug)]
pub struct DagBuilder {
    nodes: Vec<DagNode>,
    depths: Vec<u32>,
    outputs: Vec<(String, NodeId)>,
    max_depth: u32,
    max_nodes: usize,
}

impl DagBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(
            BudgetLimits::preset_default().max_recursion_depth,
            DEFAULT_MAX_NODES,
        )
    }

    #[must_use]
    pub fn with_budget(budget: &Budget) -> Self {
        Self::with_limits(budget.limits().max_recursion_depth, DEFAULT_MAX_NODES)
    }

    #[must_use]
    pub fn with_limits(max_depth: u32, max_nodes: usize) -> Self {
        Self {
            nodes: Vec::new(),
            depths: Vec::new(),
            outputs: Vec::new(),
            max_depth,
            max_nodes,
        }
    }

    pub fn add_op<T: PullOp + 'static>(&mut self, op: T) -> FfpResult<NodeId> {
        self.add_erased(erase_op(op))
    }

    pub fn add_erased(&mut self, op: Arc<dyn ErasedPullOp>) -> FfpResult<NodeId> {
        if self.nodes.len() >= self.max_nodes {
            return Err(FfpError::budget_exceeded(
                BudgetKind::RecursionDepth,
                self.max_nodes as u64,
                (self.nodes.len() + 1) as u64,
            ));
        }

        let inputs = op.inputs().to_vec();
        let mut depth = 0u32;
        for &inp in &inputs {
            let Some(&inp_depth) = self.depths.get(inp.index()) else {
                return Err(FfpError::invalid_argument(format!(
                    "node references non-existent input {inp}"
                )));
            };
            let next_depth = inp_depth
                .checked_add(1)
                .ok_or_else(|| FfpError::overflow("DAG depth overflow"))?;
            if next_depth > depth {
                depth = next_depth;
            }
        }

        if depth > self.max_depth {
            return Err(FfpError::budget_exceeded(
                BudgetKind::RecursionDepth,
                u64::from(self.max_depth),
                u64::from(depth),
            ));
        }

        let raw_id = u32::try_from(self.nodes.len())
            .map_err(|_| FfpError::overflow("DAG node id exceeds u32::MAX"))?;
        let id = NodeId(raw_id);
        self.nodes.push(DagNode { id, op, inputs });
        self.depths.push(depth);
        Ok(id)
    }

    pub fn mark_output(&mut self, name: impl Into<String>, id: NodeId) -> FfpResult<()> {
        if id.index() >= self.nodes.len() {
            return Err(FfpError::invalid_argument(format!(
                "cannot mark unknown {id} as output"
            )));
        }
        let name = name.into();
        if name.is_empty() {
            return Err(FfpError::invalid_argument("output name must not be empty"));
        }
        if let Some((_, existing)) = self.outputs.iter_mut().find(|(n, _)| n == &name) {
            *existing = id;
        } else {
            self.outputs.push((name, id));
        }
        Ok(())
    }

    #[must_use]
    pub fn build(self) -> DagGraph {
        DagGraph {
            nodes: self.nodes,
            outputs: self.outputs,
        }
    }
}

impl Default for DagBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{EvalContext, UpstreamPuller};
    use ffpix_core::{PixelFormat, Rect, StatusCode, TileViewMut};

    #[derive(Debug)]
    struct DummyOp {
        desc: ImageDescriptor,
        inputs: Vec<NodeId>,
        tag: &'static str,
        param: u32,
        access: AccessPattern,
    }

    impl DummyOp {
        fn source(tag: &'static str, param: u32, access: AccessPattern) -> Self {
            Self {
                desc: ImageDescriptor::new(64, 64, PixelFormat::RGBA8),
                inputs: Vec::new(),
                tag,
                param,
                access,
            }
        }

        fn stage(tag: &'static str, param: u32, inputs: Vec<NodeId>) -> Self {
            Self {
                desc: ImageDescriptor::new(64, 64, PixelFormat::RGBA8),
                inputs,
                tag,
                param,
                access: AccessPattern::Random,
            }
        }
    }

    impl PullOp for DummyOp {
        type ThreadState = ();

        fn descriptor(&self) -> &ImageDescriptor {
            &self.desc
        }

        fn access_pattern(&self) -> AccessPattern {
            self.access
        }

        fn inputs(&self) -> &[NodeId] {
            &self.inputs
        }

        fn start(&self, _ctx: &EvalContext) -> FfpResult<()> {
            Ok(())
        }

        fn generate(
            &self,
            _state: &mut (),
            _puller: &mut dyn UpstreamPuller,
            _req: Rect,
            _out: &mut TileViewMut<'_>,
        ) -> FfpResult<()> {
            Ok(())
        }

        fn op_kind(&self) -> OpKind {
            if self.inputs.is_empty() {
                OpKind::Source
            } else {
                OpKind::Point
            }
        }

        fn name(&self) -> &'static str {
            self.tag
        }

        fn fingerprint(&self, hasher: &mut Sha256) {
            hasher.update(self.tag.as_bytes());
            hasher.update(&self.param.to_le_bytes());
        }
    }

    #[test]
    fn linear_chain_and_diamond_topology_with_named_outputs() -> FfpResult<()> {
        let mut b = DagBuilder::new();
        let src = b.add_op(DummyOp::source("src", 1, AccessPattern::Sequential))?;
        let left = b.add_op(DummyOp::stage("blur", 2, vec![src]))?;
        let right = b.add_op(DummyOp::stage("sharpen", 3, vec![src]))?;
        let merge = b.add_op(DummyOp::stage("blend", 4, vec![left, right]))?;
        b.mark_output("main", merge)?;
        b.mark_output("preview", left)?;

        let graph = b.build();
        assert_eq!(graph.len(), 4);
        assert_eq!(graph.consumers(src), vec![left, right]);
        assert_eq!(graph.output("main"), Some(merge));
        assert_eq!(graph.output("preview"), Some(left));
        assert_eq!(
            graph.effective_access_pattern(merge)?,
            AccessPattern::Sequential
        );

        let topo = graph.topological_order()?;
        assert_eq!(topo, vec![src, left, right, merge]);
        Ok(())
    }

    #[test]
    fn invalid_node_id_and_depth_limit_rejected() {
        let mut b = DagBuilder::with_limits(2, 10);
        // References a NodeId that does not exist yet.
        let bad = b.add_op(DummyOp::stage("bad", 0, vec![NodeId(0)]));
        assert_eq!(
            bad.err().map(|e| e.status_code()),
            Some(StatusCode::InvalidArgument)
        );

        let Ok(n0) = b.add_op(DummyOp::source("n0", 0, AccessPattern::Random)) else {
            return;
        };
        let Ok(n1) = b.add_op(DummyOp::stage("n1", 1, vec![n0])) else {
            return;
        };
        let Ok(n2) = b.add_op(DummyOp::stage("n2", 2, vec![n1])) else {
            return;
        };
        // Depth 3 exceeds max_depth 2.
        let n3 = b.add_op(DummyOp::stage("n3", 3, vec![n2]));
        assert!(matches!(
            n3,
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::RecursionDepth,
                limit: 2,
                requested: 3,
            })
        ));

        assert!(b.mark_output("bad", NodeId(99)).is_err());
        assert!(b.mark_output("", n0).is_err());
    }

    #[test]
    fn deterministic_graph_hash_changes_on_param_or_edge_change() -> FfpResult<()> {
        let build_one = |param: u32| -> FfpResult<[u8; 32]> {
            let mut b = DagBuilder::new();
            let s = b.add_op(DummyOp::source("src", 1, AccessPattern::Random))?;
            let p = b.add_op(DummyOp::stage("gain", param, vec![s]))?;
            b.mark_output("output", p)?;
            Ok(b.build().graph_hash())
        };

        let h1 = build_one(10)?;
        let h2 = build_one(10)?;
        let h3 = build_one(11)?;
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        Ok(())
    }
}
