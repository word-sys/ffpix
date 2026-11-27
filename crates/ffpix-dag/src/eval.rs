//! Single-threaded pull evaluator and RAII session lifecycle management.

use ffpix_core::{FfpError, FfpResult, Rect, TileBuffer, TileViewMut};

use crate::graph::DagGraph;
use crate::op::{ErasedState, EvalContext, NodeId, UpstreamPuller};
use crate::validate::{ValidationReport, validate_graph, validate_subgraph};

/// Single-worker pull execution session.
///
/// Calls `start()` once per node in topological order, evaluates tile requests
/// on demand via `pull_rect()`, and guarantees `stop()` runs on every
/// initialized node in reverse order on both `finish()` and `Drop`.
#[derive(Debug)]
pub struct PullSession<'g> {
    graph: &'g DagGraph,
    ctx: EvalContext,
    states: Vec<Option<ErasedState>>,
    init_order: Vec<NodeId>,
    depth: u32,
    finished: bool,
}

impl<'g> PullSession<'g> {
    pub fn start(graph: &'g DagGraph, ctx: EvalContext) -> FfpResult<Self> {
        let report = validate_graph(graph, ctx.budget())?;
        Self::start_with_report(graph, ctx, &report)
    }

    pub fn start_subgraph(graph: &'g DagGraph, root: NodeId, ctx: EvalContext) -> FfpResult<Self> {
        let report = validate_subgraph(graph, root, ctx.budget())?;
        Self::start_with_report(graph, ctx, &report)
    }

    fn start_with_report(
        graph: &'g DagGraph,
        ctx: EvalContext,
        report: &ValidationReport,
    ) -> FfpResult<Self> {
        let mut states: Vec<Option<ErasedState>> = Vec::with_capacity(graph.len());
        states.resize_with(graph.len(), || None);

        let mut session = Self {
            graph,
            ctx,
            states,
            init_order: Vec::with_capacity(report.topo_order.len()),
            depth: 0,
            finished: false,
        };

        for &id in &report.topo_order {
            let node = graph.node(id)?;
            let state = node.op.start_erased(&session.ctx)?;
            session.states[id.index()] = Some(state);
            session.init_order.push(id);
        }

        Ok(session)
    }

    #[must_use]
    pub const fn graph(&self) -> &'g DagGraph {
        self.graph
    }

    #[must_use]
    pub const fn context(&self) -> &EvalContext {
        &self.ctx
    }

    pub fn pull_rect(
        &mut self,
        node_id: NodeId,
        req: Rect,
        out: &mut TileViewMut<'_>,
    ) -> FfpResult<()> {
        if self.finished {
            return Err(FfpError::invalid_argument(
                "cannot pull from a finished PullSession",
            ));
        }

        let node = self.graph.node(node_id)?;
        let desc = node.descriptor();
        req.validate()?;

        if out.format() != desc.pixel_format {
            return Err(FfpError::format_mismatch(format!(
                "pull output format {} does not match {} format {}",
                out.format(),
                node_id,
                desc.pixel_format
            )));
        }
        if out.rect().size() != req.size() {
            return Err(FfpError::dimension_mismatch(
                req.w,
                req.h,
                out.rect().w,
                out.rect().h,
            ));
        }
        if !req.is_empty() && !desc.bounds().contains_rect(req) {
            return Err(FfpError::invalid_argument(format!(
                "pull rect {req} exceeds {} bounds {}",
                node_id,
                desc.bounds()
            )));
        }
        if req.is_empty() {
            return Ok(());
        }

        self.ctx.budget().check_recursion_depth(self.depth)?;

        let idx = node_id.index();
        let mut state = self
            .states
            .get_mut(idx)
            .and_then(Option::take)
            .ok_or_else(|| {
                FfpError::invalid_argument(format!(
                    "{node_id} is not initialized or was re-entered"
                ))
            })?;

        self.depth = self
            .depth
            .checked_add(1)
            .ok_or_else(|| FfpError::overflow("pull recursion depth overflow"))?;

        let mut puller = SessionPuller { session: self };
        let res = node.op.generate_erased(&mut state, &mut puller, req, out);

        self.depth -= 1;
        // Restore state before propagating any error so stop() still runs during teardown.
        self.states[idx] = Some(state);
        res
    }

    pub fn pull_tile(&mut self, node_id: NodeId, rect: Rect) -> FfpResult<TileBuffer> {
        let fmt = self.graph.node(node_id)?.descriptor().pixel_format;
        let mut buf = self.ctx.pool().acquire(rect, fmt)?;
        self.pull_rect(node_id, rect, &mut buf.as_view_mut())?;
        Ok(buf)
    }

    /// Stops all initialized nodes in reverse topological order and returns the
    /// first error encountered, if any.
    pub fn finish(mut self) -> FfpResult<()> {
        let res = self.stop_all();
        self.finished = true;
        res
    }

    fn stop_all(&mut self) -> FfpResult<()> {
        let mut first_err: Option<FfpError> = None;
        while let Some(id) = self.init_order.pop() {
            if let Some(state) = self.states.get_mut(id.index()).and_then(Option::take)
                && let Ok(node) = self.graph.node(id)
                && let Err(err) = node.op.stop_erased(state)
                && first_err.is_none()
            {
                first_err = Some(err);
            }
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl Drop for PullSession<'_> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.stop_all();
            self.finished = true;
        }
    }
}

struct SessionPuller<'a, 'g> {
    session: &'a mut PullSession<'g>,
}

impl UpstreamPuller for SessionPuller<'_, '_> {
    fn eval_ctx(&self) -> &EvalContext {
        &self.session.ctx
    }

    fn pull_into(&mut self, node: NodeId, rect: Rect, out: &mut TileViewMut<'_>) -> FfpResult<()> {
        self.session.pull_rect(node, rect, out)
    }

    fn pull_owned(&mut self, node: NodeId, rect: Rect) -> FfpResult<TileBuffer> {
        self.session.pull_tile(node, rect)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DagBuilder;
    use crate::op::{OpKind, PullOp};
    use ffpix_core::{
        Budget, BudgetKind, BudgetLimits, ImageDescriptor, MemoryReservation, PixelFormat, Sha256,
        StatusCode, SynthPattern, fill_tile_pattern,
    };
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Debug)]
    struct LifecycleTracker {
        starts: Mutex<Vec<&'static str>>,
        stops: Mutex<Vec<&'static str>>,
        source_pulls: AtomicUsize,
    }

    impl LifecycleTracker {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                starts: Mutex::new(Vec::new()),
                stops: Mutex::new(Vec::new()),
                source_pulls: AtomicUsize::new(0),
            })
        }

        fn starts(&self) -> Vec<&'static str> {
            self.starts.lock().map(|g| g.clone()).unwrap_or_default()
        }

        fn stops(&self) -> Vec<&'static str> {
            self.stops.lock().map(|g| g.clone()).unwrap_or_default()
        }
    }

    #[derive(Debug)]
    struct PatternSource {
        tag: &'static str,
        desc: ImageDescriptor,
        pattern: SynthPattern,
        fail_on_y: Option<u32>,
        fail_on_start: bool,
        tracker: Arc<LifecycleTracker>,
    }

    struct ScratchState {
        _res: MemoryReservation,
    }

    impl PullOp for PatternSource {
        type ThreadState = ScratchState;

        fn descriptor(&self) -> &ImageDescriptor {
            &self.desc
        }

        fn start(&self, ctx: &EvalContext) -> FfpResult<Self::ThreadState> {
            if self.fail_on_start {
                return Err(FfpError::internal("injected start failure"));
            }
            if let Ok(mut g) = self.tracker.starts.lock() {
                g.push(self.tag);
            }
            Ok(ScratchState {
                _res: ctx.budget().reserve_memory(512)?,
            })
        }

        fn generate(
            &self,
            _state: &mut Self::ThreadState,
            _puller: &mut dyn UpstreamPuller,
            req: Rect,
            out: &mut TileViewMut<'_>,
        ) -> FfpResult<()> {
            self.tracker.source_pulls.fetch_add(1, Ordering::SeqCst);
            if let Some(bad_y) = self.fail_on_y
                && req.y <= bad_y
                && req.y + req.h > bad_y
            {
                return Err(FfpError::corrupt_data("injected mid-tile read failure"));
            }
            fill_tile_pattern(out, self.desc.size(), self.pattern)
        }

        fn stop(&self, state: Self::ThreadState) {
            if let Ok(mut g) = self.tracker.stops.lock() {
                g.push(self.tag);
            }
            drop(state);
        }

        fn op_kind(&self) -> OpKind {
            OpKind::Source
        }

        fn name(&self) -> &'static str {
            self.tag
        }

        fn fingerprint(&self, hasher: &mut Sha256) {
            hasher.update(self.tag.as_bytes());
        }
    }

    #[derive(Debug)]
    struct AddOffsetOp {
        tag: &'static str,
        desc: ImageDescriptor,
        inputs: Vec<NodeId>,
        delta: u8,
        tracker: Arc<LifecycleTracker>,
    }

    impl PullOp for AddOffsetOp {
        type ThreadState = ScratchState;

        fn descriptor(&self) -> &ImageDescriptor {
            &self.desc
        }

        fn inputs(&self) -> &[NodeId] {
            &self.inputs
        }

        fn start(&self, ctx: &EvalContext) -> FfpResult<Self::ThreadState> {
            if let Ok(mut g) = self.tracker.starts.lock() {
                g.push(self.tag);
            }
            Ok(ScratchState {
                _res: ctx.budget().reserve_memory(256)?,
            })
        }

        fn generate(
            &self,
            _state: &mut Self::ThreadState,
            puller: &mut dyn UpstreamPuller,
            req: Rect,
            out: &mut TileViewMut<'_>,
        ) -> FfpResult<()> {
            let src = puller.pull_owned(self.inputs[0], req)?;
            for y in 0..req.h {
                let srow = src.row(y)?;
                let drow = out.row_mut(y)?;
                for (d, &s) in drow.iter_mut().zip(srow.iter()) {
                    *d = s.wrapping_add(self.delta);
                }
            }
            puller.eval_ctx().pool().release(src);
            Ok(())
        }

        fn stop(&self, state: Self::ThreadState) {
            if let Ok(mut g) = self.tracker.stops.lock() {
                g.push(self.tag);
            }
            drop(state);
        }

        fn op_kind(&self) -> OpKind {
            OpKind::Point
        }

        fn name(&self) -> &'static str {
            self.tag
        }

        fn fingerprint(&self, hasher: &mut Sha256) {
            hasher.update(self.tag.as_bytes());
            hasher.update(&[self.delta]);
        }
    }

    #[derive(Debug)]
    struct AvgBlendOp {
        tag: &'static str,
        desc: ImageDescriptor,
        inputs: Vec<NodeId>,
        tracker: Arc<LifecycleTracker>,
    }

    impl PullOp for AvgBlendOp {
        type ThreadState = ScratchState;

        fn descriptor(&self) -> &ImageDescriptor {
            &self.desc
        }

        fn inputs(&self) -> &[NodeId] {
            &self.inputs
        }

        fn start(&self, ctx: &EvalContext) -> FfpResult<Self::ThreadState> {
            if let Ok(mut g) = self.tracker.starts.lock() {
                g.push(self.tag);
            }
            Ok(ScratchState {
                _res: ctx.budget().reserve_memory(256)?,
            })
        }

        fn generate(
            &self,
            _state: &mut Self::ThreadState,
            puller: &mut dyn UpstreamPuller,
            req: Rect,
            out: &mut TileViewMut<'_>,
        ) -> FfpResult<()> {
            let a = puller.pull_owned(self.inputs[0], req)?;
            let b = puller.pull_owned(self.inputs[1], req)?;
            for y in 0..req.h {
                let ra = a.row(y)?;
                let rb = b.row(y)?;
                let ro = out.row_mut(y)?;
                for ((dst, &va), &vb) in ro.iter_mut().zip(ra.iter()).zip(rb.iter()) {
                    *dst = ((u16::from(va) + u16::from(vb)) / 2) as u8;
                }
            }
            puller.eval_ctx().pool().release(a);
            puller.eval_ctx().pool().release(b);
            Ok(())
        }

        fn stop(&self, state: Self::ThreadState) {
            if let Ok(mut g) = self.tracker.stops.lock() {
                g.push(self.tag);
            }
            drop(state);
        }

        fn op_kind(&self) -> OpKind {
            OpKind::Composite
        }

        fn name(&self) -> &'static str {
            self.tag
        }

        fn fingerprint(&self, hasher: &mut Sha256) {
            hasher.update(self.tag.as_bytes());
        }
    }

    #[test]
    fn multi_node_chain_and_diamond_subgraph_eval_and_teardown() -> FfpResult<()> {
        let budget = Budget::preset_default();
        let tracker = LifecycleTracker::new();
        let desc = ImageDescriptor::new(32, 32, PixelFormat::RGBA8);

        let mut b = DagBuilder::with_budget(&budget);
        let src = b.add_op(PatternSource {
            tag: "src",
            desc: desc.clone(),
            pattern: SynthPattern::Solid([10, 20, 30, 40]),
            fail_on_y: None,
            fail_on_start: false,
            tracker: Arc::clone(&tracker),
        })?;
        let left = b.add_op(AddOffsetOp {
            tag: "left",
            desc: desc.clone(),
            inputs: vec![src],
            delta: 10,
            tracker: Arc::clone(&tracker),
        })?;
        let right = b.add_op(AddOffsetOp {
            tag: "right",
            desc: desc.clone(),
            inputs: vec![src],
            delta: 30,
            tracker: Arc::clone(&tracker),
        })?;
        let merge = b.add_op(AvgBlendOp {
            tag: "merge",
            desc: desc.clone(),
            inputs: vec![left, right],
            tracker: Arc::clone(&tracker),
        })?;
        b.mark_output("output", merge)?;
        let graph = b.build();

        let pool = ffpix_core::TilePool::with_defaults(budget.clone());
        let ctx = EvalContext::new(budget.clone(), pool.clone(), 0);
        let mut session = PullSession::start(&graph, ctx)?;

        // start() runs exactly once per node in topological order.
        assert_eq!(tracker.starts(), vec!["src", "left", "right", "merge"]);
        assert!(tracker.stops().is_empty());

        for y in [0u32, 16u32] {
            let tile = session.pull_tile(merge, Rect::new(0, y, 16, 16))?;
            for r in 0..16 {
                for px in tile.row(r)?.chunks_exact(4) {
                    // avg(s + 10, s + 30) = s + 20 -> [30, 40, 50, 60]
                    assert_eq!(px, &[30, 40, 50, 60]);
                }
            }
            pool.release(tile);
        }

        // Pool reused slabs across tiles and branches.
        assert!(pool.stats().reuses > 0);

        session.finish()?;
        pool.clear();

        // stop() runs in exact reverse topological order.
        assert_eq!(tracker.stops(), vec!["merge", "right", "left", "src"]);
        assert_eq!(budget.current_memory_bytes(), 0);
        Ok(())
    }

    #[test]
    fn mid_tile_error_triggers_deterministic_raii_stop_and_zero_leak() -> FfpResult<()> {
        let budget = Budget::preset_default();
        let tracker = LifecycleTracker::new();
        let desc = ImageDescriptor::new(32, 32, PixelFormat::RGBA8);

        let mut b = DagBuilder::with_budget(&budget);
        let src = b.add_op(PatternSource {
            tag: "src",
            desc: desc.clone(),
            pattern: SynthPattern::HorizontalGradient,
            fail_on_y: Some(20),
            fail_on_start: false,
            tracker: Arc::clone(&tracker),
        })?;
        let add = b.add_op(AddOffsetOp {
            tag: "add",
            desc,
            inputs: vec![src],
            delta: 1,
            tracker: Arc::clone(&tracker),
        })?;
        b.mark_output("output", add)?;
        let graph = b.build();

        {
            let ctx = EvalContext::with_budget(budget.clone(), 0);
            let mut session = PullSession::start(&graph, ctx)?;

            // Top tile succeeds.
            let ok_tile = session.pull_tile(add, Rect::new(0, 0, 16, 16))?;
            drop(ok_tile);

            // Second tile hits injected failure at y = 20.
            let err = session.pull_tile(add, Rect::new(0, 16, 16, 16));
            assert_eq!(
                err.err().map(|e| e.status_code()),
                Some(StatusCode::CorruptData)
            );
            // Session dropped here without finish(): RAII guard must still call stop()!
        }

        assert_eq!(tracker.starts(), vec!["src", "add"]);
        assert_eq!(tracker.stops(), vec!["add", "src"]);
        assert_eq!(budget.current_memory_bytes(), 0);
        Ok(())
    }

    #[test]
    fn partial_start_failure_and_recursion_depth_guard_clean_up_all_states() -> FfpResult<()> {
        let budget = Budget::new(BudgetLimits {
            max_recursion_depth: 2,
            ..BudgetLimits::preset_default()
        });
        let tracker = LifecycleTracker::new();
        let desc = ImageDescriptor::new(16, 16, PixelFormat::RGBA8);

        // Partial start failure: node 0 succeeds, node 1 fails. Node 0 must be stopped!
        let mut b1 = DagBuilder::with_budget(&budget);
        let n0 = b1.add_op(PatternSource {
            tag: "n0",
            desc: desc.clone(),
            pattern: SynthPattern::Solid([0, 0, 0, 255]),
            fail_on_y: None,
            fail_on_start: false,
            tracker: Arc::clone(&tracker),
        })?;
        let _n1 = b1.add_op(PatternSource {
            tag: "n1_fail",
            desc: desc.clone(),
            pattern: SynthPattern::Solid([0, 0, 0, 255]),
            fail_on_y: None,
            fail_on_start: true,
            tracker: Arc::clone(&tracker),
        })?;
        let g1 = b1.build();

        let res = PullSession::start(&g1, EvalContext::with_budget(budget.clone(), 0));
        assert!(res.is_err());
        assert_eq!(tracker.starts(), vec!["n0"]);
        assert_eq!(tracker.stops(), vec!["n0"]);
        assert_eq!(budget.current_memory_bytes(), 0);

        // start_subgraph initializes only reachable nodes (skipping failing n1_fail).
        let sub_session =
            PullSession::start_subgraph(&g1, n0, EvalContext::with_budget(budget.clone(), 0))?;
        sub_session.finish()?;
        assert_eq!(budget.current_memory_bytes(), 0);

        // Depth limit check: chain depth 2 succeeds with max_recursion_depth = 2.
        let mut b2 = DagBuilder::new();
        let s = b2.add_op(PatternSource {
            tag: "s",
            desc: desc.clone(),
            pattern: SynthPattern::Solid([1, 2, 3, 4]),
            fail_on_y: None,
            fail_on_start: false,
            tracker: Arc::clone(&tracker),
        })?;
        let p1 = b2.add_op(AddOffsetOp {
            tag: "p1",
            desc: desc.clone(),
            inputs: vec![s],
            delta: 1,
            tracker: Arc::clone(&tracker),
        })?;
        let p2 = b2.add_op(AddOffsetOp {
            tag: "p2",
            desc,
            inputs: vec![p1],
            delta: 1,
            tracker: Arc::clone(&tracker),
        })?;
        let g2 = b2.build();

        let mut session = PullSession::start(&g2, EvalContext::with_budget(budget.clone(), 0))?;
        let tile = session.pull_tile(p2, Rect::new(0, 0, 8, 8))?;
        drop(tile);
        session.finish()?;
        assert_eq!(budget.current_memory_bytes(), 0);

        // Tighter runtime budget with max_recursion_depth = 1 rejects depth-2 chain cleanly.
        let tight = Budget::new(BudgetLimits {
            max_recursion_depth: 1,
            ..BudgetLimits::preset_default()
        });
        assert!(matches!(
            PullSession::start(&g2, EvalContext::with_budget(tight, 0)),
            Err(FfpError::BudgetExceeded {
                kind: BudgetKind::RecursionDepth,
                ..
            })
        ));

        Ok(())
    }
}
