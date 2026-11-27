//! Iterative DAG validation, cycle detection, depth enforcement, and descriptor checks.

use ffpix_core::{Budget, FfpError, FfpResult, Rect};

use crate::graph::{DagGraph, DagNode};
use crate::op::{AccessPattern, NodeId, OpKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub node_count: usize,
    pub max_depth: u32,
    pub topo_order: Vec<NodeId>,
    pub effective_access: AccessPattern,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Color {
    White,
    Gray,
    Black,
}

/// Iterative 3-color DFS detecting any cycle without using call-stack recursion.
pub fn detect_cycles(graph: &DagGraph) -> FfpResult<()> {
    let n = graph.len();
    let mut color = vec![Color::White; n];
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for start in 0..n {
        if color[start] != Color::White {
            continue;
        }
        color[start] = Color::Gray;
        stack.push((start, 0));

        while let Some((u, edge_idx)) = stack.pop() {
            let node = graph.node(NodeId(u as u32))?;
            if edge_idx < node.inputs.len() {
                stack.push((u, edge_idx + 1));
                let inp = node.inputs[edge_idx];
                let v = inp.index();
                if v >= n {
                    return Err(FfpError::invalid_argument(format!(
                        "{} references unknown input {inp}",
                        node.id
                    )));
                }
                match color[v] {
                    Color::Gray => {
                        return Err(FfpError::cycle_detected(format!(
                            "cycle detected involving {} -> {inp}",
                            node.id
                        )));
                    }
                    Color::White => {
                        color[v] = Color::Gray;
                        stack.push((v, 0));
                    }
                    Color::Black => {}
                }
            } else {
                color[u] = Color::Black;
            }
        }
    }
    Ok(())
}

pub fn validate_graph(graph: &DagGraph, budget: &Budget) -> FfpResult<ValidationReport> {
    if graph.is_empty() {
        return Err(FfpError::invalid_input("DAG contains no nodes"));
    }
    for (idx, node) in graph.nodes().iter().enumerate() {
        let expected = u32::try_from(idx)
            .map_err(|_| FfpError::overflow("DAG node index exceeds u32::MAX"))?;
        if node.id != NodeId(expected) {
            return Err(FfpError::invalid_input(format!(
                "DAG node at slot {idx} has mismatched id {}",
                node.id
            )));
        }
    }
    for (name, id) in graph.outputs() {
        if name.is_empty() {
            return Err(FfpError::invalid_argument(
                "DAG contains an output with an empty name",
            ));
        }
        let _ = graph.node(*id)?;
    }

    let roots: Vec<usize> = (0..graph.len()).collect();
    validate_roots(graph, &roots, budget)
}

pub fn validate_subgraph(
    graph: &DagGraph,
    root: NodeId,
    budget: &Budget,
) -> FfpResult<ValidationReport> {
    let _ = graph.node(root)?;
    validate_roots(graph, &[root.index()], budget)
}

fn validate_roots(
    graph: &DagGraph,
    roots: &[usize],
    budget: &Budget,
) -> FfpResult<ValidationReport> {
    let n = graph.len();
    let mut color = vec![Color::White; n];
    let mut depth = vec![0u32; n];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut topo_order: Vec<NodeId> = Vec::new();
    let mut max_depth = 0u32;
    let mut effective_access = AccessPattern::Random;

    for &start in roots {
        if color[start] != Color::White {
            continue;
        }
        color[start] = Color::Gray;
        stack.push((start, 0));

        while let Some((u, edge_idx)) = stack.pop() {
            let node = &graph.nodes()[u];
            if edge_idx < node.inputs.len() {
                stack.push((u, edge_idx + 1));
                let inp = node.inputs[edge_idx];
                let v = inp.index();
                if v >= n {
                    return Err(FfpError::invalid_argument(format!(
                        "{} references unknown input {inp}",
                        node.id
                    )));
                }
                match color[v] {
                    Color::Gray => {
                        return Err(FfpError::cycle_detected(format!(
                            "cycle detected involving {} -> {inp}",
                            node.id
                        )));
                    }
                    Color::White => {
                        color[v] = Color::Gray;
                        stack.push((v, 0));
                    }
                    Color::Black => {}
                }
            } else {
                // All upstream inputs of `u` are Black; compute depth and validate node.
                let mut node_depth = 0u32;
                for &inp in &node.inputs {
                    let candidate = depth[inp.index()]
                        .checked_add(1)
                        .ok_or_else(|| FfpError::overflow("DAG depth overflow"))?;
                    if candidate > node_depth {
                        node_depth = candidate;
                    }
                }
                budget.check_recursion_depth(node_depth)?;
                depth[u] = node_depth;
                if node_depth > max_depth {
                    max_depth = node_depth;
                }

                validate_node(graph, node, budget)?;
                effective_access = effective_access.combine(node.access_pattern());

                color[u] = Color::Black;
                topo_order.push(node.id);
            }
        }
    }

    Ok(ValidationReport {
        node_count: topo_order.len(),
        max_depth,
        topo_order,
        effective_access,
    })
}

fn validate_node(graph: &DagGraph, node: &DagNode, budget: &Budget) -> FfpResult<()> {
    let desc = node.descriptor();
    desc.validate()?;
    budget.check_dimensions(desc.width, desc.height)?;
    budget.check_frames(desc.frame_count)?;

    match node.op_kind() {
        OpKind::Source => {
            if !node.inputs.is_empty() {
                return Err(FfpError::invalid_argument(format!(
                    "source {} must not have upstream inputs",
                    node.id
                )));
            }
        }
        OpKind::Cache => {
            if node.inputs.len() != 1 {
                return Err(FfpError::invalid_argument(format!(
                    "cache {} requires exactly 1 input, got {}",
                    node.id,
                    node.inputs.len()
                )));
            }
            let inp_desc = graph.node(node.inputs[0])?.descriptor();
            if desc.width != inp_desc.width || desc.height != inp_desc.height {
                return Err(FfpError::dimension_mismatch(
                    inp_desc.width,
                    inp_desc.height,
                    desc.width,
                    desc.height,
                ));
            }
            if desc.pixel_format != inp_desc.pixel_format || desc.alpha_mode != inp_desc.alpha_mode
            {
                return Err(FfpError::format_mismatch(format!(
                    "cache {} format/alpha mismatch with input {}",
                    node.id, node.inputs[0]
                )));
            }
        }
        OpKind::Geometry | OpKind::Resample => {
            if node.inputs.len() != 1 {
                return Err(FfpError::invalid_argument(format!(
                    "{} ({:?}) requires exactly 1 input, got {}",
                    node.id,
                    node.op_kind(),
                    node.inputs.len()
                )));
            }
            let inp_desc = graph.node(node.inputs[0])?.descriptor();
            if desc.pixel_format != inp_desc.pixel_format || desc.alpha_mode != inp_desc.alpha_mode
            {
                return Err(FfpError::format_mismatch(format!(
                    "{} ({:?}) output format {} ({}) incompatible with input {} ({})",
                    node.id,
                    node.op_kind(),
                    desc.pixel_format,
                    desc.alpha_mode,
                    inp_desc.pixel_format,
                    inp_desc.alpha_mode
                )));
            }
        }
        OpKind::Filter => {
            if node.inputs.is_empty() {
                return Err(FfpError::invalid_argument(format!(
                    "filter {} requires at least 1 input",
                    node.id
                )));
            }
            let inp_desc = graph.node(node.inputs[0])?.descriptor();
            if desc.pixel_format.sample != inp_desc.pixel_format.sample {
                return Err(FfpError::format_mismatch(format!(
                    "filter {} sample type {} mismatches input {}",
                    node.id, desc.pixel_format.sample, inp_desc.pixel_format.sample
                )));
            }
        }
        OpKind::Composite => {
            if node.inputs.len() < 2 {
                return Err(FfpError::invalid_argument(format!(
                    "composite {} requires at least 2 inputs, got {}",
                    node.id,
                    node.inputs.len()
                )));
            }
            let base_sample = graph.node(node.inputs[0])?.descriptor().pixel_format.sample;
            for &inp in &node.inputs[1..] {
                let s = graph.node(inp)?.descriptor().pixel_format.sample;
                if s != base_sample {
                    return Err(FfpError::format_mismatch(format!(
                        "composite {} input {inp} sample type {s} mismatches base {base_sample}",
                        node.id
                    )));
                }
            }
        }
        OpKind::Point | OpKind::Overlay | OpKind::Custom => {}
    }

    // Probe required_input_rect on a 1x1 origin rect to catch broken input coordinate mappings early.
    let probe = Rect::new(0, 0, desc.width.min(1), desc.height.min(1));
    for (idx, &inp) in node.inputs.iter().enumerate() {
        let req_in = node.op.required_input_rect(idx, probe)?;
        req_in.validate()?;
        let inp_bounds = graph.node(inp)?.descriptor().bounds();
        if !req_in.is_empty() && !inp_bounds.contains_rect(req_in) {
            return Err(FfpError::invalid_argument(format!(
                "{} requested out-of-bounds rect {req_in} from {inp} ({inp_bounds})",
                node.id
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DagBuilder;
    use crate::op::{EvalContext, PullOp, UpstreamPuller, erase_op};
    use ffpix_core::{
        BudgetKind, BudgetLimits, ImageDescriptor, PixelFormat, Sha256, StatusCode, TileViewMut,
    };

    #[derive(Debug)]
    struct MockNodeOp {
        desc: ImageDescriptor,
        inputs: Vec<NodeId>,
        kind: OpKind,
    }

    impl PullOp for MockNodeOp {
        type ThreadState = ();

        fn descriptor(&self) -> &ImageDescriptor {
            &self.desc
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
            self.kind
        }

        fn name(&self) -> &'static str {
            "mock"
        }

        fn fingerprint(&self, hasher: &mut Sha256) {
            hasher.update(b"mock");
        }
    }

    fn make_raw_node(id: u32, inputs: Vec<NodeId>, kind: OpKind, w: u32, h: u32) -> DagNode {
        let op = erase_op(MockNodeOp {
            desc: ImageDescriptor::new(w, h, PixelFormat::RGBA8),
            inputs: inputs.clone(),
            kind,
        });
        DagNode {
            id: NodeId(id),
            op,
            inputs,
        }
    }

    #[test]
    fn detects_self_loop_and_multi_node_cycles() {
        let budget = Budget::preset_default();

        // Self-loop: node 0 -> node 0.
        let self_loop = DagGraph::from_parts_unvalidated(
            vec![make_raw_node(0, vec![NodeId(0)], OpKind::Point, 32, 32)],
            vec![("out".to_string(), NodeId(0))],
        );
        assert_eq!(
            detect_cycles(&self_loop).err().map(|e| e.status_code()),
            Some(StatusCode::CycleDetected)
        );
        assert_eq!(
            validate_graph(&self_loop, &budget)
                .err()
                .map(|e| e.status_code()),
            Some(StatusCode::CycleDetected)
        );

        // Multi-node 3-cycle: 0 -> 1 -> 2 -> 0.
        let tri_cycle = DagGraph::from_parts_unvalidated(
            vec![
                make_raw_node(0, vec![NodeId(2)], OpKind::Point, 32, 32),
                make_raw_node(1, vec![NodeId(0)], OpKind::Point, 32, 32),
                make_raw_node(2, vec![NodeId(1)], OpKind::Point, 32, 32),
            ],
            vec![("out".to_string(), NodeId(2))],
        );
        assert_eq!(
            validate_graph(&tri_cycle, &budget)
                .err()
                .map(|e| e.status_code()),
            Some(StatusCode::CycleDetected)
        );
    }

    #[test]
    fn iterative_dfs_handles_100k_node_chain_without_stack_overflow() {
        let len = 100_000u32;
        let mut nodes = Vec::with_capacity(len as usize);
        nodes.push(make_raw_node(0, Vec::new(), OpKind::Source, 16, 16));
        for i in 1..len {
            nodes.push(make_raw_node(i, vec![NodeId(i - 1)], OpKind::Point, 16, 16));
        }
        let graph =
            DagGraph::from_parts_unvalidated(nodes, vec![("out".to_string(), NodeId(len - 1))]);

        // Strict budget rejects depth cleanly without stack overflow.
        let strict = Budget::preset_strict();
        assert!(matches!(
            validate_graph(&graph, &strict),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::RecursionDepth,
                ..
            })
        ));

        // High depth budget validates all 100,000 nodes iteratively.
        let high_depth = Budget::new(BudgetLimits {
            max_recursion_depth: len,
            ..BudgetLimits::preset_default()
        });
        let report = validate_graph(&graph, &high_depth);
        assert!(report.is_ok());
        if let Ok(rep) = report {
            assert_eq!(rep.node_count, len as usize);
            assert_eq!(rep.max_depth, len - 1);
            assert_eq!(rep.topo_order.first(), Some(&NodeId(0)));
            assert_eq!(rep.topo_order.last(), Some(&NodeId(len - 1)));
        }
    }

    #[test]
    fn rejects_intermediate_upscale_exceeding_dimension_budget() -> FfpResult<()> {
        let budget = Budget::new(BudgetLimits {
            max_width: 4096,
            max_height: 4096,
            max_pixels: 4_000_000,
            ..BudgetLimits::preset_default()
        });

        let mut b = DagBuilder::with_budget(&budget);
        let src = b.add_op(MockNodeOp {
            desc: ImageDescriptor::new(100, 100, PixelFormat::RGBA8),
            inputs: Vec::new(),
            kind: OpKind::Source,
        })?;
        // Intermediate node upscales to 3000x3000 = 9M pixels (> 4M limit) even though final output shrinks back.
        let huge = b.add_op(MockNodeOp {
            desc: ImageDescriptor::new(3000, 3000, PixelFormat::RGBA8),
            inputs: vec![src],
            kind: OpKind::Resample,
        })?;
        let small = b.add_op(MockNodeOp {
            desc: ImageDescriptor::new(200, 200, PixelFormat::RGBA8),
            inputs: vec![huge],
            kind: OpKind::Resample,
        })?;
        b.mark_output("output", small)?;

        let graph = b.build();
        assert!(matches!(
            validate_subgraph(&graph, small, &budget),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::Pixels,
                limit: 4_000_000,
                requested: 9_000_000,
            })
        ));
        Ok(())
    }

    #[test]
    fn rejects_format_and_channel_mismatches_between_connected_nodes() -> FfpResult<()> {
        let budget = Budget::preset_default();
        let mut b = DagBuilder::with_budget(&budget);
        let src = b.add_op(MockNodeOp {
            desc: ImageDescriptor::new(64, 64, PixelFormat::RGB8),
            inputs: Vec::new(),
            kind: OpKind::Source,
        })?;
        // Geometry op cannot silently change RGB8 -> RGBA16.
        let bad_geom = b.add_op(MockNodeOp {
            desc: ImageDescriptor::new(32, 32, PixelFormat::RGBA16),
            inputs: vec![src],
            kind: OpKind::Geometry,
        })?;

        let graph = b.build();
        assert_eq!(
            validate_subgraph(&graph, bad_geom, &budget)
                .err()
                .map(|e| e.status_code()),
            Some(StatusCode::FormatMismatch)
        );
        Ok(())
    }
}
