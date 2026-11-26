//! Core demand-driven `PullOp` protocol and type-erased execution adapter.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use ffpix_core::{
    Budget, FfpError, FfpResult, ImageDescriptor, Rect, Sha256, TileBuffer, TilePool, TileViewMut,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node#{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AccessPattern {
    #[default]
    Random,
    Sequential,
    SequentialUnbuffered,
}

impl AccessPattern {
    #[must_use]
    pub const fn is_sequential(self) -> bool {
        !matches!(self, Self::Random)
    }

    #[must_use]
    pub const fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::SequentialUnbuffered, _) | (_, Self::SequentialUnbuffered) => {
                Self::SequentialUnbuffered
            }
            (Self::Sequential, _) | (_, Self::Sequential) => Self::Sequential,
            (Self::Random, Self::Random) => Self::Random,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    Source,
    Point,
    Geometry,
    Resample,
    Filter,
    Composite,
    Cache,
    Overlay,
    Custom,
}

#[derive(Debug, Clone)]
pub struct EvalContext {
    budget: Budget,
    pool: TilePool,
    worker_index: usize,
}

impl EvalContext {
    #[must_use]
    pub const fn new(budget: Budget, pool: TilePool, worker_index: usize) -> Self {
        Self {
            budget,
            pool,
            worker_index,
        }
    }

    #[must_use]
    pub fn with_budget(budget: Budget, worker_index: usize) -> Self {
        let pool = TilePool::with_defaults(budget.clone());
        Self::new(budget, pool, worker_index)
    }

    #[must_use]
    pub const fn budget(&self) -> &Budget {
        &self.budget
    }

    #[must_use]
    pub const fn pool(&self) -> &TilePool {
        &self.pool
    }

    #[must_use]
    pub const fn worker_index(&self) -> usize {
        self.worker_index
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PullRequest {
    pub node_id: NodeId,
    pub rect: Rect,
}

impl PullRequest {
    #[must_use]
    pub const fn new(node_id: NodeId, rect: Rect) -> Self {
        Self { node_id, rect }
    }
}

/// Callback interface allowing a `PullOp` to request upstream rectangles during `generate`.
pub trait UpstreamPuller {
    fn eval_ctx(&self) -> &EvalContext;
    fn pull_into(&mut self, node: NodeId, rect: Rect, out: &mut TileViewMut<'_>) -> FfpResult<()>;
    fn pull_owned(&mut self, node: NodeId, rect: Rect) -> FfpResult<TileBuffer>;
}

/// Puller for standalone source evaluation where no upstream nodes exist.
#[derive(Debug)]
pub struct NoopPuller {
    ctx: EvalContext,
}

impl NoopPuller {
    #[must_use]
    pub const fn new(ctx: EvalContext) -> Self {
        Self { ctx }
    }
}

impl UpstreamPuller for NoopPuller {
    fn eval_ctx(&self) -> &EvalContext {
        &self.ctx
    }

    fn pull_into(
        &mut self,
        node: NodeId,
        _rect: Rect,
        _out: &mut TileViewMut<'_>,
    ) -> FfpResult<()> {
        Err(FfpError::invalid_argument(format!(
            "cannot pull upstream {node} from a source-only context"
        )))
    }

    fn pull_owned(&mut self, node: NodeId, _rect: Rect) -> FfpResult<TileBuffer> {
        Err(FfpError::invalid_argument(format!(
            "cannot pull upstream {node} from a source-only context"
        )))
    }
}

/// Immutable demand-driven operator node.
///
/// Mutable scratch buffers belong strictly inside `ThreadState`, created once
/// per worker thread in `start` and released in `stop`.
pub trait PullOp: Send + Sync + fmt::Debug {
    type ThreadState: Send + 'static;

    fn descriptor(&self) -> &ImageDescriptor;

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Random
    }

    fn inputs(&self) -> &[NodeId] {
        &[]
    }

    fn required_input_rect(&self, _input_index: usize, req: Rect) -> FfpResult<Rect> {
        Ok(req)
    }

    fn start(&self, ctx: &EvalContext) -> FfpResult<Self::ThreadState>;

    fn generate(
        &self,
        state: &mut Self::ThreadState,
        puller: &mut dyn UpstreamPuller,
        req: Rect,
        out: &mut TileViewMut<'_>,
    ) -> FfpResult<()>;

    fn stop(&self, state: Self::ThreadState) {
        drop(state);
    }

    fn op_kind(&self) -> OpKind {
        OpKind::Custom
    }

    fn name(&self) -> &'static str;

    fn fingerprint(&self, hasher: &mut Sha256);
}

pub struct ErasedState(Box<dyn Any + Send>);

impl fmt::Debug for ErasedState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ErasedState(..)")
    }
}

/// Object-safe wrapper around [`PullOp`] for heterogeneous DAG execution.
pub trait ErasedPullOp: Send + Sync + fmt::Debug {
    fn descriptor(&self) -> &ImageDescriptor;
    fn access_pattern(&self) -> AccessPattern;
    fn inputs(&self) -> &[NodeId];
    fn required_input_rect(&self, input_index: usize, req: Rect) -> FfpResult<Rect>;
    fn start_erased(&self, ctx: &EvalContext) -> FfpResult<ErasedState>;
    fn generate_erased(
        &self,
        state: &mut ErasedState,
        puller: &mut dyn UpstreamPuller,
        req: Rect,
        out: &mut TileViewMut<'_>,
    ) -> FfpResult<()>;
    fn stop_erased(&self, state: ErasedState) -> FfpResult<()>;
    fn op_kind(&self) -> OpKind;
    fn name(&self) -> &'static str;
    fn fingerprint(&self, hasher: &mut Sha256);
    fn as_any(&self) -> &(dyn Any + Send + Sync);
}

#[derive(Debug)]
pub struct PullOpAdapter<T: PullOp>(pub T);

impl<T: PullOp + 'static> ErasedPullOp for PullOpAdapter<T> {
    fn descriptor(&self) -> &ImageDescriptor {
        self.0.descriptor()
    }

    fn access_pattern(&self) -> AccessPattern {
        self.0.access_pattern()
    }

    fn inputs(&self) -> &[NodeId] {
        self.0.inputs()
    }

    fn required_input_rect(&self, input_index: usize, req: Rect) -> FfpResult<Rect> {
        self.0.required_input_rect(input_index, req)
    }

    fn start_erased(&self, ctx: &EvalContext) -> FfpResult<ErasedState> {
        let state = self.0.start(ctx)?;
        Ok(ErasedState(Box::new(state)))
    }

    fn generate_erased(
        &self,
        state: &mut ErasedState,
        puller: &mut dyn UpstreamPuller,
        req: Rect,
        out: &mut TileViewMut<'_>,
    ) -> FfpResult<()> {
        let typed = state
            .0
            .downcast_mut::<T::ThreadState>()
            .ok_or_else(|| FfpError::internal("mismatched PullOp ThreadState type in generate"))?;
        self.0.generate(typed, puller, req, out)
    }

    fn stop_erased(&self, state: ErasedState) -> FfpResult<()> {
        let typed = state
            .0
            .downcast::<T::ThreadState>()
            .map_err(|_| FfpError::internal("mismatched PullOp ThreadState type in stop"))?;
        self.0.stop(*typed);
        Ok(())
    }

    fn op_kind(&self) -> OpKind {
        self.0.op_kind()
    }

    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn fingerprint(&self, hasher: &mut Sha256) {
        self.0.fingerprint(hasher);
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        &self.0
    }
}

#[must_use]
pub fn erase_op<T: PullOp + 'static>(op: T) -> Arc<dyn ErasedPullOp> {
    Arc::new(PullOpAdapter(op))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffpix_core::{MemoryReservation, PixelFormat};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct ConstantOp {
        desc: ImageDescriptor,
        rgba: [u8; 4],
        stop_counter: Arc<AtomicUsize>,
    }

    struct ConstantState {
        scratch_row: Vec<u8>,
        _reservation: MemoryReservation,
    }

    impl PullOp for ConstantOp {
        type ThreadState = ConstantState;

        fn descriptor(&self) -> &ImageDescriptor {
            &self.desc
        }

        fn start(&self, ctx: &EvalContext) -> FfpResult<Self::ThreadState> {
            let row_bytes = self.desc.pixel_format.checked_row_bytes(self.desc.width)?;
            let reservation = ctx.budget().reserve_memory(row_bytes as u64)?;
            let mut scratch_row = vec![0u8; row_bytes];
            for px in scratch_row.chunks_exact_mut(4) {
                px.copy_from_slice(&self.rgba);
            }
            Ok(ConstantState {
                scratch_row,
                _reservation: reservation,
            })
        }

        fn generate(
            &self,
            state: &mut Self::ThreadState,
            _puller: &mut dyn UpstreamPuller,
            req: Rect,
            out: &mut TileViewMut<'_>,
        ) -> FfpResult<()> {
            let needed = self.desc.pixel_format.checked_row_bytes(req.w)?;
            let src = state
                .scratch_row
                .get(..needed)
                .ok_or_else(|| FfpError::invalid_argument("request width exceeds image width"))?;
            for y in 0..req.h {
                out.row_mut(y)?.copy_from_slice(src);
            }
            Ok(())
        }

        fn stop(&self, state: Self::ThreadState) {
            self.stop_counter.fetch_add(1, Ordering::SeqCst);
            drop(state);
        }

        fn op_kind(&self) -> OpKind {
            OpKind::Source
        }

        fn name(&self) -> &'static str {
            "constant"
        }

        fn fingerprint(&self, hasher: &mut Sha256) {
            hasher.update(self.name().as_bytes());
            hasher.update(&self.rgba);
        }
    }

    #[test]
    fn constant_op_lifecycle_and_erased_adapter_produce_exact_pixels() -> FfpResult<()> {
        let budget = Budget::preset_default();
        let stop_counter = Arc::new(AtomicUsize::new(0));
        let op = ConstantOp {
            desc: ImageDescriptor::new(32, 24, PixelFormat::RGBA8),
            rgba: [10, 20, 30, 255],
            stop_counter: Arc::clone(&stop_counter),
        };
        let erased = erase_op(op);

        assert_eq!(erased.op_kind(), OpKind::Source);
        assert_eq!(erased.access_pattern(), AccessPattern::Random);
        assert!(erased.inputs().is_empty());
        assert!(erased.as_any().downcast_ref::<ConstantOp>().is_some());

        let ctx = EvalContext::with_budget(budget.clone(), 0);
        let mut puller = NoopPuller::new(ctx.clone());

        let mut state = erased.start_erased(&ctx)?;
        assert_eq!(budget.current_memory_bytes(), 32 * 4);

        let req = Rect::new(4, 6, 8, 5);
        let mut tile = TileBuffer::new_zeroed(req, PixelFormat::RGBA8, &budget)?;
        erased.generate_erased(&mut state, &mut puller, req, &mut tile.as_view_mut())?;

        for y in 0..req.h {
            let row = tile.row(y)?;
            for px in row.chunks_exact(4) {
                assert_eq!(px, &[10, 20, 30, 255]);
            }
        }

        erased.stop_erased(state)?;
        assert_eq!(stop_counter.load(Ordering::SeqCst), 1);

        drop(tile);
        assert_eq!(budget.current_memory_bytes(), 0);
        Ok(())
    }

    #[test]
    fn access_pattern_combination_and_fingerprint_determinism() {
        assert_eq!(
            AccessPattern::Random.combine(AccessPattern::Random),
            AccessPattern::Random
        );
        assert_eq!(
            AccessPattern::Random.combine(AccessPattern::Sequential),
            AccessPattern::Sequential
        );
        assert_eq!(
            AccessPattern::Sequential.combine(AccessPattern::SequentialUnbuffered),
            AccessPattern::SequentialUnbuffered
        );
        assert!(!AccessPattern::Random.is_sequential());
        assert!(AccessPattern::Sequential.is_sequential());

        let op = ConstantOp {
            desc: ImageDescriptor::new(8, 8, PixelFormat::RGBA8),
            rgba: [1, 2, 3, 4],
            stop_counter: Arc::new(AtomicUsize::new(0)),
        };
        let mut h1 = Sha256::new();
        let mut h2 = Sha256::new();
        op.fingerprint(&mut h1);
        op.fingerprint(&mut h2);
        assert_eq!(h1.finalize(), h2.finalize());
    }
}
