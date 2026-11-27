//! Thread-safe bounded LRU tile cache keyed by `(NodeId, Rect)`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ffpix_core::{FfpResult, Rect, TileBuffer};

use crate::op::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCacheKey {
    pub node_id: NodeId,
    pub rect: Rect,
}

impl TileCacheKey {
    #[must_use]
    pub const fn new(node_id: NodeId, rect: Rect) -> Self {
        Self { node_id, rect }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TileCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: usize,
    pub used_bytes: u64,
    pub max_bytes: u64,
}

#[derive(Debug)]
struct LruNode {
    key: TileCacheKey,
    tile: Arc<TileBuffer>,
    bytes: u64,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug)]
struct CacheInner {
    map: HashMap<TileCacheKey, usize>,
    nodes: Vec<Option<LruNode>>,
    free_slots: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
    used_bytes: u64,
    max_bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl CacheInner {
    fn new(max_bytes: u64) -> Self {
        Self {
            map: HashMap::new(),
            nodes: Vec::new(),
            free_slots: Vec::new(),
            head: None,
            tail: None,
            used_bytes: 0,
            max_bytes,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = match self.nodes.get(idx).and_then(Option::as_ref) {
            Some(node) => (node.prev, node.next),
            None => return,
        };
        match prev {
            Some(p) => {
                if let Some(pnode) = self.nodes.get_mut(p).and_then(Option::as_mut) {
                    pnode.next = next;
                }
            }
            None => self.head = next,
        }
        match next {
            Some(n) => {
                if let Some(nnode) = self.nodes.get_mut(n).and_then(Option::as_mut) {
                    nnode.prev = prev;
                }
            }
            None => self.tail = prev,
        }
        if let Some(node) = self.nodes.get_mut(idx).and_then(Option::as_mut) {
            node.prev = None;
            node.next = None;
        }
    }

    fn push_front(&mut self, idx: usize) {
        let old_head = self.head;
        if let Some(node) = self.nodes.get_mut(idx).and_then(Option::as_mut) {
            node.prev = None;
            node.next = old_head;
        }
        match old_head {
            Some(h) => {
                if let Some(hnode) = self.nodes.get_mut(h).and_then(Option::as_mut) {
                    hnode.prev = Some(idx);
                }
            }
            None => self.tail = Some(idx),
        }
        self.head = Some(idx);
    }

    fn move_to_front(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }
        self.detach(idx);
        self.push_front(idx);
    }

    fn remove_slot(&mut self, idx: usize) -> Option<Arc<TileBuffer>> {
        self.detach(idx);
        let node = self.nodes.get_mut(idx).and_then(Option::take)?;
        self.map.remove(&node.key);
        self.used_bytes = self.used_bytes.saturating_sub(node.bytes);
        self.free_slots.push(idx);
        Some(node.tile)
    }

    fn pop_tail(&mut self) -> Option<Arc<TileBuffer>> {
        let tail_idx = self.tail?;
        self.remove_slot(tail_idx)
    }

    fn alloc_slot(&mut self, node: LruNode) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            self.nodes[slot] = Some(node);
            slot
        } else {
            let slot = self.nodes.len();
            self.nodes.push(Some(node));
            slot
        }
    }
}

#[derive(Debug, Clone)]
pub struct TileCache {
    inner: Arc<Mutex<CacheInner>>,
}

impl TileCache {
    #[must_use]
    pub fn new(max_bytes: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheInner::new(max_bytes))),
        }
    }

    #[must_use]
    pub fn get(&self, node_id: NodeId, rect: Rect) -> Option<Arc<TileBuffer>> {
        let key = TileCacheKey::new(node_id, rect);
        let mut guard = self.inner.lock().ok()?;
        if let Some(&idx) = guard.map.get(&key) {
            guard.move_to_front(idx);
            guard.hits += 1;
            return guard
                .nodes
                .get(idx)
                .and_then(Option::as_ref)
                .map(|n| Arc::clone(&n.tile));
        }
        guard.misses += 1;
        None
    }

    /// Inserts `tile` into the cache, evicting LRU entries as needed to stay
    /// within `max_bytes`. Returns `false` if the tile alone exceeds `max_bytes`.
    pub fn insert(&self, node_id: NodeId, rect: Rect, tile: Arc<TileBuffer>) -> bool {
        let key = TileCacheKey::new(node_id, rect);
        let bytes = tile_tracked_bytes(&tile);
        let mut dropped: Vec<Arc<TileBuffer>> = Vec::new();

        let inserted = if let Ok(mut guard) = self.inner.lock() {
            if let Some(&existing_idx) = guard.map.get(&key)
                && let Some(old) = guard.remove_slot(existing_idx)
            {
                dropped.push(old);
            }

            if guard.max_bytes == 0 || bytes > guard.max_bytes {
                false
            } else {
                while guard.used_bytes.saturating_add(bytes) > guard.max_bytes {
                    if let Some(evicted) = guard.pop_tail() {
                        guard.evictions += 1;
                        dropped.push(evicted);
                    } else {
                        break;
                    }
                }
                let node = LruNode {
                    key,
                    tile,
                    bytes,
                    prev: None,
                    next: None,
                };
                let slot = guard.alloc_slot(node);
                guard.push_front(slot);
                guard.map.insert(key, slot);
                guard.used_bytes = guard.used_bytes.saturating_add(bytes);
                true
            }
        } else {
            false
        };

        // Drop evicted tiles outside the mutex lock so Budget memory releases immediately.
        drop(dropped);
        inserted
    }

    pub fn get_or_insert_with<F>(
        &self,
        node_id: NodeId,
        rect: Rect,
        produce: F,
    ) -> FfpResult<Arc<TileBuffer>>
    where
        F: FnOnce() -> FfpResult<TileBuffer>,
    {
        if let Some(hit) = self.get(node_id, rect) {
            return Ok(hit);
        }
        let tile = Arc::new(produce()?);
        let _ = self.insert(node_id, rect, Arc::clone(&tile));
        Ok(tile)
    }

    pub fn remove(&self, node_id: NodeId, rect: Rect) -> Option<Arc<TileBuffer>> {
        let key = TileCacheKey::new(node_id, rect);
        let mut guard = self.inner.lock().ok()?;
        let idx = *guard.map.get(&key)?;
        guard.remove_slot(idx)
    }

    pub fn clear(&self) {
        let mut drained: Vec<Arc<TileBuffer>> = Vec::new();
        if let Ok(mut guard) = self.inner.lock() {
            for node in guard.nodes.drain(..).flatten() {
                drained.push(node.tile);
            }
            guard.map.clear();
            guard.free_slots.clear();
            guard.head = None;
            guard.tail = None;
            guard.used_bytes = 0;
        }
        drop(drained);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.map.len()).unwrap_or(0)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.inner.lock().map(|g| g.used_bytes).unwrap_or(0)
    }

    #[must_use]
    pub fn max_bytes(&self) -> u64 {
        self.inner.lock().map(|g| g.max_bytes).unwrap_or(0)
    }

    #[must_use]
    pub fn stats(&self) -> TileCacheStats {
        match self.inner.lock() {
            Ok(g) => TileCacheStats {
                hits: g.hits,
                misses: g.misses,
                evictions: g.evictions,
                entries: g.map.len(),
                used_bytes: g.used_bytes,
                max_bytes: g.max_bytes,
            },
            Err(_) => TileCacheStats::default(),
        }
    }
}

fn tile_tracked_bytes(tile: &TileBuffer) -> u64 {
    let res = tile.reserved_bytes();
    if res > 0 {
        res
    } else {
        tile.data().len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffpix_core::{Budget, PixelFormat};
    use std::thread;

    fn make_tile(
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        tag: u8,
        budget: &Budget,
    ) -> FfpResult<Arc<TileBuffer>> {
        let mut buf = TileBuffer::new_zeroed(Rect::new(x, y, w, h), PixelFormat::RGBA8, budget)?;
        buf.data_mut().fill(tag);
        Ok(Arc::new(buf))
    }

    #[test]
    fn lru_eviction_order_and_telemetry_counters() -> FfpResult<()> {
        let budget = Budget::preset_default();
        // Each 16x16 RGBA8 tile is 1024 bytes; cap holds exactly 2 tiles.
        let cache = TileCache::new(2048);
        let n = NodeId(0);
        let r0 = Rect::new(0, 0, 16, 16);
        let r1 = Rect::new(16, 0, 16, 16);
        let r2 = Rect::new(32, 0, 16, 16);

        assert!(cache.insert(n, r0, make_tile(0, 0, 16, 16, 1, &budget)?));
        assert!(cache.insert(n, r1, make_tile(16, 0, 16, 16, 2, &budget)?));
        assert_eq!(cache.used_bytes(), 2048);
        assert_eq!(budget.current_memory_bytes(), 2048);

        // Touch r0 so r1 becomes the LRU victim.
        let hit0 = cache.get(n, r0);
        assert!(hit0.is_some());
        if let Some(t) = hit0 {
            assert_eq!(t.data()[0], 1);
            drop(t);
        }

        // Inserting r2 must evict r1 and keep r0.
        assert!(cache.insert(n, r2, make_tile(32, 0, 16, 16, 3, &budget)?));
        assert_eq!(cache.used_bytes(), 2048);
        assert_eq!(budget.current_memory_bytes(), 2048);
        assert!(cache.get(n, r1).is_none());
        assert!(cache.get(n, r0).is_some());
        assert!(cache.get(n, r2).is_some());

        let stats = cache.stats();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.used_bytes, 2048);

        // Oversized tile (> max_bytes) is rejected without evicting existing entries.
        let huge = make_tile(0, 0, 32, 32, 9, &budget)?; // 4096 bytes
        assert!(!cache.insert(NodeId(1), Rect::new(0, 0, 32, 32), huge));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.used_bytes(), 2048);

        // Clear immediately drops tiles and releases all Budget memory.
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.used_bytes(), 0);
        assert_eq!(budget.current_memory_bytes(), 0);
        Ok(())
    }

    #[test]
    fn drop_cache_releases_all_budget_memory() -> FfpResult<()> {
        let budget = Budget::preset_default();
        {
            let cache = TileCache::new(4096);
            for i in 0..4u32 {
                let r = Rect::new(i * 16, 0, 16, 16);
                assert!(cache.insert(
                    NodeId(0),
                    r,
                    make_tile(i * 16, 0, 16, 16, i as u8, &budget)?
                ));
            }
            assert_eq!(budget.current_memory_bytes(), 4096);
        }
        assert_eq!(budget.current_memory_bytes(), 0);
        Ok(())
    }

    #[test]
    fn concurrent_reads_and_writes_across_4_threads_stay_bounded() {
        let budget = Budget::preset_default();
        let cap = 8 * 1024u64; // holds at most 8 tiles of 1024 bytes
        let cache = TileCache::new(cap);

        let mut handles = Vec::new();
        for tid in 0..4u32 {
            let c = cache.clone();
            let b = budget.clone();
            handles.push(thread::spawn(move || {
                for iter in 0..100u32 {
                    let tile_idx = (tid * 7 + iter) % 24;
                    let rect = Rect::new(tile_idx * 16, 0, 16, 16);
                    let res = c.get_or_insert_with(NodeId(0), rect, || {
                        let mut buf = TileBuffer::new_zeroed(rect, PixelFormat::RGBA8, &b)?;
                        buf.data_mut().fill(tile_idx as u8);
                        Ok(buf)
                    });
                    assert!(res.is_ok());
                    if let Ok(tile) = res {
                        assert_eq!(tile.data()[0], tile_idx as u8);
                    }
                    assert!(c.used_bytes() <= cap);
                }
            }));
        }

        for h in handles {
            assert!(h.join().is_ok());
        }

        let stats = cache.stats();
        assert!(stats.used_bytes <= cap);
        assert!(stats.hits > 0);
        assert!(stats.evictions > 0);

        cache.clear();
        assert_eq!(budget.current_memory_bytes(), 0);
    }
}
