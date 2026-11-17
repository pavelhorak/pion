import sys, tempfile, asyncio
import pionlite as pion
d = tempfile.mkdtemp()
r = pion.Redis(dir=d, decode_responses=True)
assert r.set("a", "b") and r.get("a") == "b"
r.rpush("l", 1, 2, 3); assert r.lrange("l", 0, -1) == ["1", "2", "3"]
assert r.eval("return redis.call('get', KEYS[1])", 1, "a") == "b"
p = r.pipeline(transaction=True); p.incr("n"); p.incr("n"); assert p.execute() == [1, 2]
assert r.blpop("nothing", timeout=0.2) is None
async def a():
    ar = pion.asyncio.Redis(dir=d, decode_responses=True)
    assert await ar.get("a") == "b"
    ps = ar.pubsub(); await ps.subscribe("ch"); await ps.get_message(timeout=1)
    await ar.publish("ch", "hi")
    m = await ps.get_message(timeout=2, ignore_subscribe_messages=True)
    assert m and m["data"] == "hi", m
    await ps.aclose(); await ar.aclose()
    import redis.asyncio
    for proto in (2, 3):   # a pool built from a URL, as docket builds its own
        pool = pion.asyncio.ConnectionPool.from_url("pion://" + d, protocol=proto, decode_responses=True)
        pr = redis.asyncio.Redis(connection_pool=pool)
        assert await pr.get("a") == "b" and await pr.hset("h", "f", "v") in (0, 1)
        assert await pr.hgetall("h") == {"f": "v"}, await pr.hgetall("h")
        await pr.aclose(); await pool.aclose()
asyncio.run(a())
assert r.get("a") == "b"   # closing a pool leaves the engine open
print(sys.version.split()[0], "smoke ok", r.pion_info()["version"])
