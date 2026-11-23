/**
 * DeXposure-Claw Live Cloudflare Edge Worker + On-Demand PyTorch Container + R2 Store
 *
 * Architecture:
 * 1. Normal user traffic (`GET /`, `/assets/live-*.json`, `/api/*`) NEVER wakes the container:
 *    it reads live dynamically updated artifacts directly from Cloudflare R2 (`env.LIVE_BUCKET`)
 *    at global edge latency (falling back to bundled `env.ASSETS` before first sync).
 * 2. Scheduled Cron (`15 *\/6 * * *`) or on-demand `POST /api/refresh` wakes the singleton
 *    `DexposureEngine` Cloudflare Container (Python 3.12 + PyTorch CPU + DGL + DeXposure-FM),
 *    which fetches 1,200+ live DeFiLlama protocol token compositions, runs authentic
 *    DeXposure-FM (`h=1,4,8w`, `MC=50`) inference + Walk-Forward Backtest settlement,
 *    persists the fresh JSONs into R2, and immediately calls `this.stop()` so compute
 *    billing drops back to zero.
 */

import { Container, getContainer } from "@cloudflare/containers";

const LIVE_JSON_FILES = new Set([
  "live-network-3d.json",
  "live-horizons.json",
  "live-history.json",
  "live-contagion-graph.json"
]);

const CHECKPOINT_IDENTITY = {
  model_family: "dexposure_fm_v4",
  checkpoint_dir: "checkpoints/dexposure-fm-v4",
  checkpoint_file: "dexposure-fm-v4.pt",
  checkpoint_sha256: "fc79295b40c7cfc355b9a3708e2366ed10d8d38bd5a2a1709882af4b46c1b694",
  schema_version: 4,
  legacy_schema: false,
  settlement_basis: "constant_price_net_flow_ex_eth_btc_beta",
  training_supervision: "fractional_constant_price_net_flow_monotonic_v4",
  pi_min: 0.5
};

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type"
};

function jsonResponse(body, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store, must-revalidate",
      ...CORS_HEADERS,
      ...extraHeaders
    }
  });
}

async function readR2Json(env, key) {
  if (!env.LIVE_BUCKET) return null;
  const obj = await env.LIVE_BUCKET.get(key);
  if (!obj) return null;
  try {
    return await obj.json();
  } catch (_) {
    return null;
  }
}

async function loadLiveOrBundledJson(request, env, filename) {
  const r2Data = await readR2Json(env, `live/${filename}`);
  if (r2Data !== null) {
    return { data: r2Data, source: "cloudflare-r2-live" };
  }
  const assetUrl = new URL(`/assets/${filename}`, request.url);
  const res = await env.ASSETS.fetch(new Request(assetUrl.toString()));
  if (!res.ok) {
    throw new Error(`Failed to load ${filename} (${res.status})`);
  }
  return { data: await res.json(), source: "bundled-fallback" };
}

const SLIM_ENTRYPOINT_PY = [
  "import gc, uvicorn",
  "from dexposure_agent.fm_predictor import FMPredictor",
  "_orig_load = FMPredictor._load_model",
  "_orig_pred = FMPredictor.predict",
  "_MEMO = {}",
  "def _slim_load(self, horizon):",
  "    hk = self._get_horizon_key(horizon)",
  "    if hk not in self._models:",
  "        for a in ('_models', '_link_scorers', '_node_heads'): getattr(self, a, {}).clear()",
  "        gc.collect()",
  "    return _orig_load(self, horizon)",
  "def _memo_pred(self, graph, horizon):",
  "    key = (graph.date, len(graph.nodes), len(graph.edges), self._get_horizon_key(horizon), round(float(self.pi_min), 4))",
  "    if key in _MEMO:",
  "        snap, probs = _MEMO[key]",
  "        self.last_edge_probs = dict(probs)",
  "        return snap",
  "    snap = _orig_pred(self, graph, horizon)",
  "    if len(_MEMO) >= 10: _MEMO.clear()",
  "    _MEMO[key] = (snap, dict(self.last_edge_probs))",
  "    return snap",
  "FMPredictor._load_model = _slim_load",
  "FMPredictor.predict = _memo_pred",
  "uvicorn.run('dexposure_agent.container_service:app', host='0.0.0.0', port=8080)"
].join("\n");

export class DexposureEngine extends Container {
  defaultPort = 8080;
  sleepAfter = "15m";
  enableInternet = true;
  pingEndpoint = "ping";
  entrypoint = ["python3", "-c", SLIM_ENTRYPOINT_PY];
  envVars = {
    DEXPOSURE_CACHE_DIR: "/var/tmp/dexposure_live_cache_v3",
    MALLOC_ARENA_MAX: "2"
  };

  async onStop(params) {
    const cur = await readR2Json(this.env, "live/status.json");
    if (cur && cur.state === "running") {
      await this.env.LIVE_BUCKET.put(
        "live/status.json",
        JSON.stringify({
          ...cur,
          state: "error",
          stage: "container_stopped_unexpectedly",
          stopParams: params || null,
          finishedAt: new Date().toISOString()
        }),
        { httpMetadata: { contentType: "application/json; charset=utf-8" } }
      );
    }
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/__trigger" && request.method === "POST") {
      return await this.handleTriggerRefresh(url.searchParams.get("force") !== "false");
    }
    return await super.fetch(request);
  }

  async handleTriggerRefresh(force = true) {
    const curStatus = await readR2Json(this.env, "live/status.json");
    if (curStatus && curStatus.state === "running" && curStatus.startedAt) {
      const ageMs = Date.now() - Date.parse(curStatus.startedAt);
      if (Number.isFinite(ageMs) && ageMs >= 0 && ageMs < 8 * 60 * 1000) {
        return jsonResponse({ accepted: false, reason: "already_running", ...curStatus }, 200);
      }
    }

    const bootState = {
      state: "running",
      stage: "booting_container",
      startedAt: new Date().toISOString(),
      finishedAt: null,
      force: Boolean(force),
      bootAttempts: 0,
      error: null
    };
    await this.env.LIVE_BUCKET.put("live/status.json", JSON.stringify(bootState), {
      httpMetadata: { contentType: "application/json; charset=utf-8" }
    });
    await this.schedule(1, "kickOrPollJob", force ? "true" : "false");
    return jsonResponse({ accepted: true, ...bootState }, 202);
  }

  async kickOrPollJob(forceFlag = "true") {
    const cur = (await readR2Json(this.env, "live/status.json")) || {};
    const startedAt = cur.startedAt || new Date().toISOString();
    const attempts = Number(cur.bootAttempts || 0) + 1;

    try {
      await this.startAndWaitForPorts({
        ports: 8080,
        cancellationOptions: {
          instanceGetTimeoutMS: 45000,
          portReadyTimeoutMS: 45000,
          waitInterval: 1000
        }
      });
      const startRes = await this.containerFetch(
        `http://container/refresh/start?force=${forceFlag === "false" ? "false" : "true"}`,
        { method: "POST" },
        8080
      );
      const startBody = await startRes.json();
      await this.env.LIVE_BUCKET.put(
        "live/status.json",
        JSON.stringify({ ...startBody, startedAt }),
        { httpMetadata: { contentType: "application/json; charset=utf-8" } }
      );
      await this.schedule(8, "pollJob", "poll");
    } catch (err) {
      const msg = String(err);
      if (attempts < 6) {
        await this.env.LIVE_BUCKET.put(
          "live/status.json",
          JSON.stringify({
            state: "running",
            stage: `warming_container_attempt_${attempts}`,
            startedAt,
            bootAttempts: attempts,
            lastWarmNotice: msg,
            error: null
          }),
          { httpMetadata: { contentType: "application/json; charset=utf-8" } }
        );
        await this.schedule(10, "kickOrPollJob", forceFlag);
        return;
      }
      const failState = {
        state: "error",
        stage: "boot_failed",
        startedAt,
        finishedAt: new Date().toISOString(),
        bootAttempts: attempts,
        error: msg
      };
      await this.env.LIVE_BUCKET.put("live/status.json", JSON.stringify(failState), {
        httpMetadata: { contentType: "application/json; charset=utf-8" }
      });
      try {
        await this.stop();
      } catch (_) {}
    }
  }

  async pollJob() {
    try {
      const res = await this.containerFetch("http://container/refresh/status", { method: "GET" }, 8080);
      if (!res.ok) {
        throw new Error(`Container status HTTP ${res.status}`);
      }
      const st = await res.json();

      if (st.state === "running") {
        await this.renewActivityTimeout();
        await this.env.LIVE_BUCKET.put("live/status.json", JSON.stringify(st), {
          httpMetadata: { contentType: "application/json; charset=utf-8" }
        });
        await this.schedule(8, "pollJob", "poll");
        return;
      }

      if (st.state === "done") {
        for (const fn of LIVE_JSON_FILES) {
          const artRes = await this.containerFetch(`http://container/artifacts/${fn}`, { method: "GET" }, 8080);
          if (!artRes.ok) {
            throw new Error(`Failed harvesting ${fn}: HTTP ${artRes.status}`);
          }
          const buf = await artRes.arrayBuffer();
          await this.env.LIVE_BUCKET.put(`live/${fn}`, buf, {
            httpMetadata: { contentType: "application/json; charset=utf-8" }
          });
        }

        if (Array.isArray(st.iconFiles)) {
          for (const ico of st.iconFiles) {
            const exists = await this.env.LIVE_BUCKET.head(`icons/${ico}`);
            if (!exists) {
              const icoRes = await this.containerFetch(`http://container/artifacts/icons/${ico}`, { method: "GET" }, 8080);
              if (icoRes.ok) {
                await this.env.LIVE_BUCKET.put(`icons/${ico}`, await icoRes.arrayBuffer(), {
                  httpMetadata: { contentType: "image/webp" }
                });
              }
            }
          }
        }

        const meta = {
          snapshotDate: st.snapshotDate,
          updatedAt: st.updatedAt || new Date().toISOString(),
          elapsedSec: st.elapsedSec,
          fullNodes: st.fullNodes,
          fullEdges: st.fullEdges,
          source: "cloudflare-container-r2-live"
        };
        await this.env.LIVE_BUCKET.put("live/meta.json", JSON.stringify(meta), {
          httpMetadata: { contentType: "application/json; charset=utf-8" }
        });
        await this.env.LIVE_BUCKET.put(
          "live/status.json",
          JSON.stringify({ ...st, state: "idle", stage: "synced_to_r2" }),
          { httpMetadata: { contentType: "application/json; charset=utf-8" } }
        );
      } else {
        await this.env.LIVE_BUCKET.put("live/status.json", JSON.stringify(st), {
          httpMetadata: { contentType: "application/json; charset=utf-8" }
        });
      }
    } catch (err) {
      await this.env.LIVE_BUCKET.put(
        "live/status.json",
        JSON.stringify({
          state: "error",
          stage: "poll_or_harvest_failed",
          finishedAt: new Date().toISOString(),
          error: String(err)
        }),
        { httpMetadata: { contentType: "application/json; charset=utf-8" } }
      );
    }

    // Immediately shut down the container once done or errored so billing drops to zero
    try {
      await this.stop();
    } catch (_) {}
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    // 1. Live R2-backed JSON assets (dynamic chain data + FM forecasts)
    if (url.pathname.startsWith("/assets/") && request.method === "GET") {
      const subPath = url.pathname.slice("/assets/".length);
      if (LIVE_JSON_FILES.has(subPath)) {
        if (env.LIVE_BUCKET) {
          const r2Obj = await env.LIVE_BUCKET.get(`live/${subPath}`);
          if (r2Obj) {
            return new Response(r2Obj.body, {
              status: 200,
              headers: {
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store, must-revalidate",
                "X-Data-Source": "cloudflare-r2-live",
                ...CORS_HEADERS
              }
            });
          }
        }
        const fallbackRes = await env.ASSETS.fetch(new Request(new URL(url.pathname, request.url).toString()));
        return new Response(fallbackRes.body, {
          status: fallbackRes.status,
          headers: {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store, must-revalidate",
            "X-Data-Source": "bundled-fallback",
            ...CORS_HEADERS
          }
        });
      }

      if (subPath.startsWith("icons/")) {
        const staticIcon = await env.ASSETS.fetch(new Request(new URL(url.pathname, request.url).toString()));
        if (staticIcon.ok) {
          return staticIcon;
        }
        const icoName = subPath.slice("icons/".length);
        if (env.LIVE_BUCKET && icoName) {
          const r2Icon = await env.LIVE_BUCKET.get(`icons/${icoName}`);
          if (r2Icon) {
            return new Response(r2Icon.body, {
              status: 200,
              headers: {
                "Content-Type": "image/webp",
                "Cache-Control": "public, max-age=86400",
                ...CORS_HEADERS
              }
            });
          }
        }
      }
    }

    // 2. Live pipeline status & on-demand container refresh API
    if (url.pathname === "/api/live-status" && request.method === "GET") {
      const [meta, status] = await Promise.all([
        readR2Json(env, "live/meta.json"),
        readR2Json(env, "live/status.json")
      ]);
      return jsonResponse({
        status: "ok",
        hasLiveR2Snapshot: Boolean(meta),
        meta,
        job: status || { state: "idle", stage: "ready" }
      });
    }

    if ((url.pathname === "/api/refresh" || url.pathname === "/api/live-history/run") && request.method === "POST") {
      const engine = getContainer(env.DEXPOSURE_ENGINE, "singleton");
      const force = url.searchParams.get("force") !== "false";
      return await engine.fetch(
        new Request(`http://do/__trigger?force=${force ? "true" : "false"}`, { method: "POST" })
      );
    }

    if (url.pathname === "/health" && request.method === "GET") {
      const meta = await readR2Json(env, "live/meta.json");
      return jsonResponse({
        status: "ok",
        runtime: "cloudflare-workers-r2-plus-container",
        fm_available: true,
        available_horizons: [1, 4, 8],
        pi_min: 0.5,
        live_snapshot: meta,
        checkpoint_identity: CHECKPOINT_IDENTITY,
        timestamp: Date.now() / 1000
      });
    }

    if (url.pathname === "/api/cases" && request.method === "GET") {
      try {
        const { data: bundle, source } = await loadLiveOrBundledJson(request, env, "live-horizons.json");
        return jsonResponse({
          status: "ok",
          has_live_run: true,
          runtime_mode: source,
          latest_live: bundle["4"] || bundle["1"]
        });
      } catch (err) {
        return jsonResponse({ status: "error", message: String(err) }, 500);
      }
    }

    if (url.pathname === "/api/live-history" && request.method === "GET") {
      try {
        const { data: history } = await loadLiveOrBundledJson(request, env, "live-history.json");
        return jsonResponse(history);
      } catch (err) {
        return jsonResponse({ status: "error", message: String(err) }, 500);
      }
    }

    if ((url.pathname === "/api/live-run" || url.pathname === "/live-predict") && request.method === "POST") {
      let body = {};
      try {
        body = await request.json();
      } catch (_) {}
      const rawHorizon = String(body.horizon || "4");
      const horizonKey = ["1", "4", "8"].includes(rawHorizon) ? rawHorizon : "4";
      try {
        const { data: bundle, source } = await loadLiveOrBundledJson(request, env, "live-horizons.json");
        const selected = JSON.parse(JSON.stringify(bundle[horizonKey] || bundle["4"] || bundle["1"]));
        selected.runtime_mode = source;
        return jsonResponse(selected);
      } catch (err) {
        return jsonResponse({ status: "error", message: String(err) }, 500);
      }
    }

    if (["/dashboard", "/monitor", "/user", "/pro", "/compliance"].includes(url.pathname)) {
      const rootUrl = new URL("/", request.url);
      const assetRes = await env.ASSETS.fetch(new Request(rootUrl.toString()));
      return new Response(assetRes.body, {
        status: 200,
        headers: {
          "Content-Type": "text/html; charset=utf-8",
          "Cache-Control": "no-cache, must-revalidate"
        }
      });
    }

    return env.ASSETS.fetch(request);
  },

  async scheduled(_event, env, ctx) {
    const engine = getContainer(env.DEXPOSURE_ENGINE, "singleton");
    ctx.waitUntil(
      engine.fetch(new Request("http://do/__trigger?force=true", { method: "POST" }))
    );
  }
};
