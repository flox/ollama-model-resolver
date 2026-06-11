//! Ordered, bounded parallel map for I/O-bound per-item work — e.g. annotating
//! a page of search results, where each item makes its own network requests.
//! Uses only `std` (scoped threads); no async runtime, no extra dependencies.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Apply `work` to every item across up to `concurrency` worker threads, and
/// return the results in the original order of `items`.
///
/// Each worker builds its own `state` once via `make_state` (e.g. a per-worker
/// HTTP client/cache) and reuses it for the items it claims. Work is handed out
/// by a shared atomic index, so uneven per-item costs balance across workers
/// (no static chunking). Order is restored by tagging each result with its
/// index and sorting, so the output is deterministic and identical to a serial
/// `items.iter().map(...)`.
///
/// `work` must not panic; a worker panic propagates out of this call.
pub fn par_map<T, S, U>(
    items: &[T],
    concurrency: usize,
    make_state: impl Fn() -> S + Sync,
    work: impl Fn(&mut S, &T) -> U + Sync,
) -> Vec<U>
where
    T: Sync,
    U: Send,
{
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    let workers = concurrency.clamp(1, n);
    let next = AtomicUsize::new(0);

    let mut indexed: Vec<(usize, U)> = thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut state = make_state();
                    let mut local: Vec<(usize, U)> = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        local.push((i, work(&mut state, &items[i])));
                    }
                    local
                })
            })
            .collect();

        handles
            .into_iter()
            .flat_map(|h| h.join().expect("par_map worker panicked"))
            .collect()
    });

    indexed.sort_by_key(|(i, _)| *i);
    indexed.into_iter().map(|(_, u)| u).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_input_order() {
        let items: Vec<usize> = (0..50).collect();
        let out = par_map(&items, 8, || (), |_, &x| x * 2);
        let expected: Vec<usize> = (0..50).map(|x| x * 2).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn builds_state_at_most_concurrency_times() {
        let items: Vec<usize> = (0..20).collect();
        let states = AtomicUsize::new(0);
        let out = par_map(
            &items,
            4,
            || {
                states.fetch_add(1, Ordering::Relaxed);
            },
            |_, &x| x,
        );
        assert_eq!(out, items); // still correct + ordered
        assert!(
            states.load(Ordering::Relaxed) <= 4,
            "built {} states for concurrency 4",
            states.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn handles_empty_single_and_concurrency_above_len() {
        let empty: Vec<usize> = Vec::new();
        assert!(par_map(&empty, 8, || (), |_, &x| x).is_empty());

        let one = vec![42usize];
        assert_eq!(par_map(&one, 8, || (), |_, &x| x), vec![42]);

        let few = vec![1usize, 2, 3];
        assert_eq!(par_map(&few, 1, || (), |_, &x| x), vec![1, 2, 3]); // serial path
    }
}
