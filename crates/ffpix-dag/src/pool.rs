//! Pure-`std` fixed-size worker pool with bounded submission, work stealing,
//! cooperative early cancellation, and panic-safe RAII thread shutdown.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use ffpix_core::{FfpError, FfpResult};

pub const DEFAULT_QUEUE_MULTIPLIER: usize = 4;

type Job = Box<dyn FnOnce(usize) + Send + 'static>;

#[derive(Debug, Clone)]
pub struct CancelToken {
    cancelled: Arc<AtomicBool>,
}

impl CancelToken {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

struct QueueState {
    queue: VecDeque<Job>,
    max_capacity: usize,
    shutdown: bool,
}

struct PoolShared {
    state: Mutex<QueueState>,
    not_empty: Condvar,
    not_full: Condvar,
}

impl PoolShared {
    fn push(&self, job: Job) -> FfpResult<()> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| FfpError::internal("worker pool queue mutex poisoned"))?;
        while guard.queue.len() >= guard.max_capacity && !guard.shutdown {
            guard = self
                .not_full
                .wait(guard)
                .map_err(|_| FfpError::internal("worker pool wait mutex poisoned"))?;
        }
        if guard.shutdown {
            return Err(FfpError::internal(
                "cannot submit job to a shut-down WorkerPool",
            ));
        }
        guard.queue.push_back(job);
        self.not_empty.notify_one();
        Ok(())
    }

    fn pop(&self) -> Option<Job> {
        let mut guard = self.state.lock().ok()?;
        loop {
            if let Some(job) = guard.queue.pop_front() {
                self.not_full.notify_one();
                return Some(job);
            }
            if guard.shutdown {
                return None;
            }
            guard = self.not_empty.wait(guard).ok()?;
        }
    }

    fn shutdown(&self) {
        if let Ok(mut guard) = self.state.lock() {
            guard.shutdown = true;
            self.not_empty.notify_all();
            self.not_full.notify_all();
        }
    }
}

/// Fixed-size worker pool (`threads == 1` executes inline with zero OS threads).
pub struct WorkerPool {
    threads: usize,
    max_queue_len: usize,
    shared: Option<Arc<PoolShared>>,
    handles: Vec<JoinHandle<()>>,
    steal_count: Arc<AtomicUsize>,
}

impl std::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("threads", &self.threads)
            .field("max_queue_len", &self.max_queue_len)
            .field("os_workers", &self.handles.len())
            .finish()
    }
}

impl WorkerPool {
    #[must_use]
    pub fn default_threads() -> usize {
        thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
    }

    pub fn new(threads: usize) -> FfpResult<Self> {
        let queue_cap = threads.saturating_mul(DEFAULT_QUEUE_MULTIPLIER).max(4);
        Self::with_queue_capacity(threads, queue_cap)
    }

    pub fn with_queue_capacity(threads: usize, max_queue_len: usize) -> FfpResult<Self> {
        if threads == 0 {
            return Err(FfpError::invalid_argument(
                "worker pool thread count must be at least 1",
            ));
        }
        if max_queue_len == 0 {
            return Err(FfpError::invalid_argument(
                "worker pool queue capacity must be at least 1",
            ));
        }

        if threads == 1 {
            return Ok(Self {
                threads: 1,
                max_queue_len,
                shared: None,
                handles: Vec::new(),
                steal_count: Arc::new(AtomicUsize::new(0)),
            });
        }

        if cfg!(target_arch = "wasm32") {
            return Err(FfpError::unsupported_feature(
                "multi-threaded WorkerPool is not supported on wasm32",
            ));
        }

        let shared = Arc::new(PoolShared {
            state: Mutex::new(QueueState {
                queue: VecDeque::with_capacity(max_queue_len.min(1024)),
                max_capacity: max_queue_len,
                shutdown: false,
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
        });

        let mut handles = Vec::with_capacity(threads);
        for worker_idx in 0..threads {
            let s = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("ffpix-worker-{worker_idx}"))
                .spawn(move || {
                    while let Some(job) = s.pop() {
                        let _ = catch_unwind(AssertUnwindSafe(|| job(worker_idx)));
                    }
                })
                .map_err(|err| {
                    shared.shutdown();
                    FfpError::Io(err)
                });

            match handle {
                Ok(h) => handles.push(h),
                Err(err) => {
                    for h in handles {
                        let _ = h.join();
                    }
                    return Err(err);
                }
            }
        }

        Ok(Self {
            threads,
            max_queue_len,
            shared: Some(shared),
            handles,
            steal_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    #[must_use]
    pub const fn threads(&self) -> usize {
        self.threads
    }

    #[must_use]
    pub const fn queue_capacity(&self) -> usize {
        self.max_queue_len
    }

    #[must_use]
    pub fn os_worker_count(&self) -> usize {
        self.handles.len()
    }

    #[must_use]
    pub fn steal_count(&self) -> usize {
        self.steal_count.load(Ordering::Relaxed)
    }

    pub fn submit<F>(&self, job: F) -> FfpResult<()>
    where
        F: FnOnce(usize) + Send + 'static,
    {
        match &self.shared {
            None => {
                let res = catch_unwind(AssertUnwindSafe(|| job(0)));
                if res.is_err() {
                    return Err(FfpError::internal("worker inline task panicked"));
                }
                Ok(())
            }
            Some(shared) => shared.push(Box::new(job)),
        }
    }

    /// Executes `items` in parallel across the pool with work-stealing queues
    /// and cooperative cancellation on the first error or panic.
    pub fn execute_batch<I, F>(&self, items: Vec<I>, work: F) -> FfpResult<()>
    where
        I: Send + 'static,
        F: Fn(usize, I) -> FfpResult<()> + Send + Sync + 'static,
    {
        self.execute_batch_with_state(
            items,
            |_| Ok(()),
            move |_, w, item| work(w, item),
            |_| Ok(()),
        )
    }

    /// Executes `items` with per-worker setup/teardown state (`init_worker` and
    /// `finish_worker` are guaranteed to pair even when a task errors or panics).
    pub fn execute_batch_with_state<I, S, Init, Work, Finish>(
        &self,
        items: Vec<I>,
        init_worker: Init,
        work: Work,
        finish_worker: Finish,
    ) -> FfpResult<()>
    where
        I: Send + 'static,
        S: Send + 'static,
        Init: Fn(usize) -> FfpResult<S> + Send + Sync + 'static,
        Work: Fn(&mut S, usize, I) -> FfpResult<()> + Send + Sync + 'static,
        Finish: Fn(S) -> FfpResult<()> + Send + Sync + 'static,
    {
        if items.is_empty() {
            return Ok(());
        }

        if self.threads == 1 || items.len() == 1 {
            return run_inline(items, init_worker, work, finish_worker);
        }

        let Some(shared) = &self.shared else {
            return run_inline(items, init_worker, work, finish_worker);
        };

        let active_workers = self.threads.min(items.len());
        let mut queues: Vec<Mutex<VecDeque<I>>> = Vec::with_capacity(active_workers);
        for _ in 0..active_workers {
            queues.push(Mutex::new(VecDeque::new()));
        }
        for (idx, item) in items.into_iter().enumerate() {
            let slot = idx % active_workers;
            if let Ok(mut q) = queues[slot].lock() {
                q.push_back(item);
            }
        }

        let cancel = CancelToken::new();
        let first_err: Arc<Mutex<Option<FfpError>>> = Arc::new(Mutex::new(None));
        let remaining = Arc::new((Mutex::new(active_workers), Condvar::new()));
        let queues = Arc::new(queues);
        let init_worker = Arc::new(init_worker);
        let work = Arc::new(work);
        let finish_worker = Arc::new(finish_worker);

        for local_slot in 0..active_workers {
            let w_queues = Arc::clone(&queues);
            let w_cancel = cancel.clone();
            let w_first_err = Arc::clone(&first_err);
            let w_remaining = Arc::clone(&remaining);
            let w_steal_count = Arc::clone(&self.steal_count);
            let w_init = Arc::clone(&init_worker);
            let w_work = Arc::clone(&work);
            let w_finish = Arc::clone(&finish_worker);

            let submit_res = shared.push(Box::new(move |worker_idx| {
                let run_res = run_worker_slice(
                    local_slot,
                    worker_idx,
                    active_workers,
                    &w_queues,
                    &w_cancel,
                    &w_steal_count,
                    w_init.as_ref(),
                    w_work.as_ref(),
                    w_finish.as_ref(),
                );
                if let Err(err) = run_res {
                    w_cancel.cancel();
                    if let Ok(mut slot) = w_first_err.lock()
                        && slot.is_none()
                    {
                        *slot = Some(err);
                    }
                }
                let (lock, cv) = &*w_remaining;
                if let Ok(mut rem) = lock.lock() {
                    *rem = rem.saturating_sub(1);
                    if *rem == 0 {
                        cv.notify_all();
                    }
                }
            }));

            if let Err(err) = submit_res {
                cancel.cancel();
                let (lock, _) = &*remaining;
                if let Ok(mut rem) = lock.lock() {
                    *rem = rem.saturating_sub(active_workers - local_slot);
                }
                wait_batch(&remaining);
                return Err(err);
            }
        }

        wait_batch(&remaining);

        if let Ok(mut slot) = first_err.lock()
            && let Some(err) = slot.take()
        {
            return Err(err);
        }
        Ok(())
    }

    pub fn shutdown(&mut self) {
        if let Some(shared) = self.shared.take() {
            shared.shutdown();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn wait_batch(remaining: &(Mutex<usize>, Condvar)) {
    let (lock, cv) = remaining;
    if let Ok(mut rem) = lock.lock() {
        while *rem > 0 {
            match cv.wait(rem) {
                Ok(next) => rem = next,
                Err(_) => break,
            }
        }
    }
}

fn run_inline<I, S, Init, Work, Finish>(
    items: Vec<I>,
    init_worker: Init,
    work: Work,
    finish_worker: Finish,
) -> FfpResult<()>
where
    Init: Fn(usize) -> FfpResult<S>,
    Work: Fn(&mut S, usize, I) -> FfpResult<()>,
    Finish: Fn(S) -> FfpResult<()>,
{
    let mut state = match catch_unwind(AssertUnwindSafe(|| init_worker(0))) {
        Ok(res) => res?,
        Err(_) => return Err(FfpError::internal("worker init panicked")),
    };

    let mut run_err: Option<FfpError> = None;
    for item in items {
        let step = catch_unwind(AssertUnwindSafe(|| work(&mut state, 0, item)));
        match step {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                run_err = Some(err);
                break;
            }
            Err(_) => {
                run_err = Some(FfpError::internal("worker task panicked"));
                break;
            }
        }
    }

    let finish_res = match catch_unwind(AssertUnwindSafe(|| finish_worker(state))) {
        Ok(res) => res,
        Err(_) => Err(FfpError::internal("worker finish panicked")),
    };

    if let Some(err) = run_err {
        return Err(err);
    }
    finish_res
}

#[allow(clippy::too_many_arguments)]
fn run_worker_slice<I, S, Init, Work, Finish>(
    local_slot: usize,
    worker_idx: usize,
    active_workers: usize,
    queues: &[Mutex<VecDeque<I>>],
    cancel: &CancelToken,
    steal_count: &AtomicUsize,
    init_worker: &Init,
    work: &Work,
    finish_worker: &Finish,
) -> FfpResult<()>
where
    Init: Fn(usize) -> FfpResult<S>,
    Work: Fn(&mut S, usize, I) -> FfpResult<()>,
    Finish: Fn(S) -> FfpResult<()>,
{
    if cancel.is_cancelled() {
        return Ok(());
    }

    let mut state = match catch_unwind(AssertUnwindSafe(|| init_worker(worker_idx))) {
        Ok(res) => res?,
        Err(_) => return Err(FfpError::internal("worker init panicked")),
    };

    let mut run_err: Option<FfpError> = None;
    while !cancel.is_cancelled() {
        let item = pop_or_steal(local_slot, active_workers, queues, steal_count);
        let Some(item) = item else {
            break;
        };
        match catch_unwind(AssertUnwindSafe(|| work(&mut state, worker_idx, item))) {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                run_err = Some(err);
                cancel.cancel();
                break;
            }
            Err(_) => {
                run_err = Some(FfpError::internal("worker task panicked"));
                cancel.cancel();
                break;
            }
        }
    }

    let finish_res = match catch_unwind(AssertUnwindSafe(|| finish_worker(state))) {
        Ok(res) => res,
        Err(_) => Err(FfpError::internal("worker finish panicked")),
    };

    if let Some(err) = run_err {
        return Err(err);
    }
    finish_res
}

fn pop_or_steal<I>(
    local_slot: usize,
    active_workers: usize,
    queues: &[Mutex<VecDeque<I>>],
    steal_count: &AtomicUsize,
) -> Option<I> {
    if let Some(q) = queues.get(local_slot)
        && let Ok(mut guard) = q.lock()
        && let Some(item) = guard.pop_front()
    {
        return Some(item);
    }

    for offset in 1..active_workers {
        let victim = (local_slot + offset) % active_workers;
        if let Some(q) = queues.get(victim)
            && let Ok(mut guard) = q.lock()
            && let Some(item) = guard.pop_back()
        {
            steal_count.fetch_add(1, Ordering::Relaxed);
            return Some(item);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffpix_core::StatusCode;
    use std::time::Duration;

    #[test]
    fn executes_across_1_2_4_and_8_threads_with_exact_completion() -> FfpResult<()> {
        for &threads in &[1usize, 2, 4, 8] {
            let pool = WorkerPool::new(threads)?;
            assert_eq!(pool.threads(), threads);
            if threads == 1 {
                assert_eq!(pool.os_worker_count(), 0);
            } else {
                assert_eq!(pool.os_worker_count(), threads);
            }

            let n = 128usize;
            let results = Arc::new(Mutex::new(vec![0usize; n]));
            let items: Vec<usize> = (0..n).collect();
            let res_clone = Arc::clone(&results);

            pool.execute_batch(items, move |worker_idx, item| {
                assert!(worker_idx < threads);
                if let Ok(mut g) = res_clone.lock() {
                    g[item] = item * 3 + 7;
                }
                Ok(())
            })?;

            if let Ok(g) = results.lock() {
                for (i, &val) in g.iter().enumerate() {
                    assert_eq!(val, i * 3 + 7);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn cooperative_early_cancellation_stops_remaining_work_and_runs_finish() -> FfpResult<()> {
        for &threads in &[1usize, 4] {
            let pool = WorkerPool::new(threads)?;
            let completed = Arc::new(AtomicUsize::new(0));
            let inits = Arc::new(AtomicUsize::new(0));
            let finishes = Arc::new(AtomicUsize::new(0));

            let items: Vec<usize> = (0..200).collect();
            let c_done = Arc::clone(&completed);
            let c_init = Arc::clone(&inits);
            let c_fin = Arc::clone(&finishes);

            let res = pool.execute_batch_with_state(
                items,
                move |_| {
                    c_init.fetch_add(1, Ordering::SeqCst);
                    Ok(1u32)
                },
                move |_state, _worker, item| {
                    if item == 3 {
                        return Err(FfpError::corrupt_data("injected worker failure"));
                    }
                    if threads > 1 {
                        thread::sleep(Duration::from_millis(1));
                    }
                    c_done.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
                move |_state| {
                    c_fin.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            );

            assert_eq!(
                res.err().map(|e| e.status_code()),
                Some(StatusCode::CorruptData)
            );
            assert!(completed.load(Ordering::SeqCst) < 200);
            // Every started worker state must have its finish callback invoked.
            assert_eq!(
                inits.load(Ordering::SeqCst),
                finishes.load(Ordering::SeqCst)
            );
        }
        Ok(())
    }

    #[test]
    fn worker_panic_is_caught_and_converted_to_internal_error() -> FfpResult<()> {
        for &threads in &[1usize, 4] {
            let pool = WorkerPool::new(threads)?;
            let res = pool.execute_batch((0..16).collect(), |_, item| {
                if item == 5 {
                    let trigger = std::hint::black_box(0usize);
                    let _boom = 1usize / trigger;
                }
                Ok(())
            });
            assert_eq!(
                res.err().map(|e| e.status_code()),
                Some(StatusCode::Internal)
            );
        }
        Ok(())
    }

    #[test]
    fn work_stealing_and_repeated_create_drop_cycles_stay_clean() -> FfpResult<()> {
        // Skewed workload triggers stealing from busy workers by fast workers.
        let pool = WorkerPool::new(4)?;
        let items: Vec<usize> = (0..40).collect();
        let done = Arc::new(AtomicUsize::new(0));
        let done_c = Arc::clone(&done);
        pool.execute_batch(items, move |_, item| {
            if item % 4 == 0 {
                thread::sleep(Duration::from_millis(3));
            }
            done_c.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })?;
        assert_eq!(done.load(Ordering::Relaxed), 40);
        assert!(pool.steal_count() > 0);
        drop(pool);

        // Repeated create/drop cycles cleanly join all threads without leaking handles.
        for cycle in 0..20usize {
            let p = WorkerPool::with_queue_capacity(4, 2)?;
            let sum = Arc::new(AtomicUsize::new(0));
            let s_c = Arc::clone(&sum);
            p.execute_batch((1..=8).collect(), move |_, x| {
                s_c.fetch_add(x, Ordering::Relaxed);
                Ok(())
            })?;
            assert_eq!(sum.load(Ordering::Relaxed), 36, "cycle {cycle}");
        }

        assert!(WorkerPool::new(0).is_err());
        assert!(WorkerPool::with_queue_capacity(2, 0).is_err());
        Ok(())
    }
}
