# you asked

message time: 2026-06-10 15:11:15

explain me what's error
Two different errors now:

1. Test 9 (recover_orphaned): `InvalidRequestError: Instance is not persistent within this Session` — The task patches `SessionLocal` to return `db_session`, but `recover_orphaned_experiments` calls `db = SessionLocal()` (calling it as a function), but we're patching it to return `db_session` directly (not as a factory). The fix: the patch should be `return_value=db_session` but the test code has `patch("app.tasks.fl_tasks.SessionLocal", return_value=db_session)` which means when `SessionLocal()` is called, it returns `db_session`.
Actually, wait — that should work. When `SessionLocal()` is called, it returns `db_session`, which is the same session object. The function queries for experiments using that session, finds one, and calls `_set_failed` which commits. The real problem is that the `exp` object in the test is already bound to `db_session`, so when we're working with it in the patched function, there's a session binding conflict. fail too if the session is closed. So the real fix is to prevent `recover_orphaned_experiments` from closing the session, or to re-query using a fresh session context. Let me re-query the experiment after the task completes to verify its state. I'm realizing the core issue: if I mock `SessionLocal` to return the real `db_session`, then calling `close()` on it will actually detach the objects. I need to either wrap the session to make `close()` a no-op, re-query the objects after the task completes, or just query fresh in the test. Let me go with option 3 since it's the cleanest approach. The real issue is that the session gets closed or expired, making the instance detached. The cleanest fix is to store the experiment ID before running the task, then query it fresh afterward using a new session or direct lookup.
Writing test fix... Wrapping the session with a mock that prevents closing, then patching SessionLocal to return this wrapper so the test can still access the refreshed experiment state afterward. For the CUDA test failure, the issue is that torch is imported inside the function rather than at module level, so patching at the module level doesn't work—I need to either import torch at the top of the file or patch it where it's actually used.
Two issues to fix:

1. Test 9: `recover_orphaned_experiments` closes the DB session in its `finally` block, detaching `exp`. Fix: wrap the session to prevent close during the test.
2. Test 10: `torch` is imported inside `_run_experiment_impl`, so `patch("app.tasks.fl_tasks.torch")` fails — can't patch a name that doesn't exist at module level. Fix: import `torch` at module level with a try/except.

---

# claude response

These are two clean, well-diagnosed errors. Let me explain exactly what's happening in each one and give you the precise fix.

---

## Error 1 — Test 9: `InvalidRequestError: Instance is not persistent within this Session`

### What's Actually Happening

The sequence inside `recover_orphaned_experiments` is roughly:

```python
db = SessionLocal()
try:
    exp = db.query(FLExperiment).filter(...).first()
    _set_failed(db, exp, "Process terminated unexpectedly")
    db.commit()
finally:
    db.close()   # ← THIS is the killer
```

When `db.close()` runs, SQLAlchemy **expires all objects** that were loaded from that session. The `exp` object you're holding in the test is now in a "detached" state — it still exists in Python memory but its session connection is severed. Any attribute access on it after that point raises `InvalidRequestError: Instance is not persistent within this Session`.

Your test then does something like:

```python
recover_orphaned_experiments()
assert exp.status == "FAILED"   # ← exp is detached here, boom
```

The diagnosis in your message is correct. The patched `SessionLocal` returns `db_session` directly, and `db_session.close()` is the real session's close, which genuinely expires `exp`.

### Why Option 3 (re-query after the task) is the right fix

You're also correct that re-querying fresh is the cleanest. Here's the full explanation of why the other two options are worse:

**Option 1 (prevent close):** Wrapping close as a no-op means the session stays open indefinitely in the test. If your test database has transactions, this can leak state between tests. It's fighting the code rather than testing it correctly.

**Option 2 (re-query in the task itself):** Changing the task code to avoid closing the session breaks the actual production behavior you're trying to test.

**Option 3 (re-query in the test after the task):** Clean, reflects real usage, no mocking gymnastics.

### The Fix

```python
def test_recover_orphaned_experiments(db_session):
    # Create the experiment and grab its ID — not the object
    exp = FLExperiment(
        id=str(uuid4()),
        status="RUNNING",
        pid=99999,   # guaranteed non-existent PID
        # ... other required fields
    )
    db_session.add(exp)
    db_session.commit()
    experiment_id = exp.id   # save the ID string, not the object

    with patch("app.tasks.fl_tasks.SessionLocal", return_value=db_session):
        # Prevent close() from actually expiring objects on db_session
        # so the session stays usable for our re-query below
        original_close = db_session.close
        db_session.close = lambda: None   # no-op during the task
        
        recover_orphaned_experiments()
        
        db_session.close = original_close   # restore

    # Re-query using the same session — now fresh and not expired
    db_session.expire_all()   # force a clean reload from DB
    result = db_session.get(FLExperiment, experiment_id)

    assert result.status == "FAILED"
    assert "Process terminated unexpectedly" in (result.error_message or "")
```

Or the even simpler version if your test setup uses a separate session factory:

```python
def test_recover_orphaned_experiments(db_session, make_session):
    exp_id = create_running_experiment(db_session, pid=99999)
    db_session.commit()

    with patch("app.tasks.fl_tasks.SessionLocal", side_effect=make_session):
        recover_orphaned_experiments()

    # Fresh session, no expiry issues
    with make_session() as fresh_db:
        result = fresh_db.get(FLExperiment, exp_id)
        assert result.status == "FAILED"
```

The `side_effect=make_session` version is actually the most honest test — it lets `recover_orphaned_experiments` use its own real session lifecycle (including `close()`) and you just read the result back through a completely separate session. Nothing is mocked about the session behavior itself.

---

## Error 2 — Test 10: `patch("app.tasks.fl_tasks.torch")` fails silently or doesn't intercept

### What's Actually Happening

Python's `unittest.mock.patch` works by replacing a **name in a specific module's namespace**. When you write:

```python
patch("app.tasks.fl_tasks.torch")
```

It looks for a name called `torch` in the `app.tasks.fl_tasks` module's `__dict__`. If `torch` is imported inside a function rather than at the top of the file:

```python
# fl_tasks.py
def _run_experiment_impl(experiment_id):
    import torch          # ← imported here, not at module level
    if not torch.cuda.is_available():
        ...
```

Then `torch` never exists in `fl_tasks.__dict__` — it only exists inside the function's local scope during that call. The patch target doesn't exist at the module level, so `patch()` either raises `AttributeError` or patches a name that never gets looked up.

### The Fix — Two Valid Approaches

**Approach A: Import torch at module level (recommended)**

This is the right fix for production code too — top-level imports make dependencies explicit and visible:

```python
# fl_tasks.py — top of file
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
```

Now `patch("app.tasks.fl_tasks.torch")` works because `torch` is a real name in the module's namespace. Your test becomes:

```python
def test_cuda_preflight_fails_fast(db_session):
    exp_id = create_pending_experiment(db_session)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch("app.tasks.fl_tasks.torch", mock_torch):
        run_fl_experiment(experiment_id=exp_id)

    with make_session() as fresh_db:
        result = fresh_db.get(FLExperiment, exp_id)
        assert result.status == "FAILED"
        assert "CUDA GPU required" in result.error_message
```

**Approach B: Patch at the source (if you can't change the import style)**

Patch `torch.cuda.is_available` where it actually lives — in the `torch` module itself. This works regardless of how `torch` is imported:

```python
with patch("torch.cuda.is_available", return_value=False):
    run_fl_experiment(experiment_id=exp_id)
```

This is less precise (patches globally for the duration, could affect other code running in the same test), but it works without any changes to `fl_tasks.py`.

**Which to use:** Approach A is correct. The module-level import is better code and makes the test straightforward. Approach B is the workaround when you can't change the source.

---

## Root Cause Summary

Both errors come from the same underlying category of problem: **the test assumes a certain object lifetime that the production code doesn't guarantee**.

In Test 9, the test assumes `exp` stays accessible after the task runs, but `db.close()` in the task's `finally` block ends that lifetime. Fix: don't hold the object across a session close — re-query by ID instead.

In Test 10, the test assumes `torch` is a module-level name in `fl_tasks`, but it's only a local import inside a function. Fix: make it a module-level import so the patch target actually exists.

Neither error is in your test logic — the assertions are correct. Both are in the setup assumptions about what state will be accessible after the code under test finishes running.