"""Production FastAPI + Gradio application for the fine-tuned GGUF model."""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from html import escape

import gradio as gr
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from runtime import (
    EXPLAIN,
    SHORT,
    AdmissionController,
    InferenceService,
    InputRejected,
    LlamaServerManager,
    MetricsRegistry,
    QueueSaturated,
    QueueWaitTimeout,
    RateLimitExceeded,
    ServiceUnavailable,
    Settings,
    decode_data_uri,
    log_event,
)

SETTINGS = Settings.from_env()
METRICS = MetricsRegistry()
MANAGER = LlamaServerManager(SETTINGS, METRICS)
ADMISSION = AdmissionController(SETTINGS, METRICS)
SERVICE = InferenceService(SETTINGS, MANAGER, ADMISSION, METRICS)


class InferPayload(BaseModel):
    image_data_uri: str = Field(min_length=32)
    question: str = Field(min_length=1, max_length=SETTINGS.max_question_chars)
    response_mode: str = SHORT
    max_tokens: int = Field(default=64, ge=8, le=SETTINGS.max_output_tokens)


def _client_id(request: Request | gr.Request | None) -> str:
    if request is None:
        return "anonymous"
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return str(host or "anonymous")


def _is_admin(value: str | None) -> bool:
    return bool(
        SETTINGS.admin_token
        and value
        and hmac.compare_digest(value, SETTINGS.admin_token)
    )


def status_payload() -> dict[str, object]:
    return {
        "service": MANAGER.status(),
        "metrics": METRICS.snapshot(),
        "limits": {
            "max_queue_size": SETTINGS.max_queue_size,
            "max_question_chars": SETTINGS.max_question_chars,
            "max_output_tokens": SETTINGS.max_output_tokens,
            "rate_limit_requests": SETTINGS.rate_limit_requests,
            "rate_limit_window_s": SETTINGS.rate_limit_window_s,
        },
        "model": {
            "repo": SETTINGS.model_repo,
            "revision": MANAGER.resolved_revision,
            "file": SETTINGS.model_file,
            "mmproj": SETTINGS.mmproj_file,
        },
    }


def render_status() -> str:
    manager = MANAGER.status()
    metrics = METRICS.snapshot()
    state = str(manager["state"])
    labels = {
        "created": ("INITIALIZING", "服務正在初始化", "amber"),
        "starting": ("LOADING", "模型權重與 llama.cpp 載入中", "amber"),
        "ready": ("READY", "可接受圖表分析請求", "green"),
        "error": ("DEGRADED", "後端暫時不可用，已記錄錯誤", "red"),
        "stopped": ("STOPPED", "服務已停止", "red"),
    }
    code, description, tone = labels.get(state, (state.upper(), state, "amber"))
    return f"""
    <div class="service-strip {tone}" role="status" aria-live="polite">
      <span class="service-dot"></span>
      <span><b>{escape(code)}</b><small>{escape(description)}</small></span>
      <span class="service-stat"><b>{metrics["queued"]}</b><small>排隊中</small></span>
      <span class="service-stat"><b>{metrics["in_flight"]}</b><small>推論中</small></span>
    </div>
    """


def gradio_answer(
    image, question: str, response_mode: str, max_tokens: int, request: gr.Request
):
    if image is None:
        raise gr.Error("請先上傳一張圖表。")
    try:
        result = SERVICE.infer(
            image,
            question,
            response_mode,
            max_tokens,
            client_id=_client_id(request),
        )
        return result.answer, result.trace, render_status()
    except InputRejected as exc:
        raise gr.Error(str(exc)) from exc
    except RateLimitExceeded as exc:
        raise gr.Error(str(exc)) from exc
    except (QueueSaturated, QueueWaitTimeout, ServiceUnavailable) as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        log_event("ui_unhandled_error", error_type=type(exc).__name__)
        raise gr.Error("推論服務發生未預期錯誤；請稍後再試。") from exc


CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;600;700;800;900&family=Noto+Serif+TC:wght@700;900&display=swap');
:root { --ink:#101722; --paper:#f3f0e8; --paper2:#e7e0d0; --signal:#ff5a36; --acid:#c6f04d; --line:#263345; --ui:'Noto Sans TC','Microsoft JhengHei',sans-serif; --display:'Noto Serif TC','PMingLiU',serif; --mono:'IBM Plex Mono','Noto Sans TC',monospace; }
::selection { background:var(--acid); color:var(--ink); }
html,body { background:var(--paper) !important; }
.gradio-container { width:calc(100% - 32px) !important; max-width:1180px !important; margin:0 auto !important; background:var(--paper) !important; color:var(--ink) !important; font-family:var(--ui) !important; --body-background-fill:var(--paper) !important; --background-fill-primary:#fffdf7 !important; --background-fill-secondary:#f3f0e8 !important; --block-background-fill:#fffdf7 !important; --block-border-color:#263345 !important; --input-background-fill:#fff !important; --body-text-color:#101722 !important; --block-label-text-color:#475467 !important; --input-placeholder-color:#7b8796 !important; --button-secondary-background-fill:#fffdf7 !important; --button-secondary-text-color:#101722 !important; --button-secondary-border-color:#263345 !important; }
.gradio-container:before { content:''; position:fixed; inset:0; pointer-events:none; opacity:.24; background-image:radial-gradient(#192231 .65px,transparent .65px); background-size:10px 10px; }
.chartops-shell { border-top:8px solid var(--ink); padding:30px 2px 18px; position:relative; }
.eyebrow { font:600 13px var(--mono); letter-spacing:.12em; display:flex; justify-content:space-between; gap:20px; border-bottom:1px solid var(--line); padding-bottom:12px; }
.hero-title { color:var(--ink) !important; font-family:var(--display); font-size:clamp(48px,5vw,72px); line-height:1.04; letter-spacing:-.055em; margin:26px 0 18px; max-width:800px; font-weight:900; }
.hero-title em { color:var(--signal); font-style:normal; }
.deck { max-width:780px; font-size:17px; line-height:1.85; color:#475467; letter-spacing:.01em; }
.metric-strip { display:grid; grid-template-columns:repeat(4,1fr); border:2px solid var(--line); margin:26px 0 16px; background:#fffdf7; }
.metric { padding:14px 16px; border-right:1px solid var(--line); } .metric:last-child{border:0}
.metric b { display:block; font:600 12px var(--mono); letter-spacing:.08em; color:#667085; }
.metric span { display:block; margin-top:6px; color:var(--ink)!important; font-size:17px; font-weight:800; }
.scope-note { border-left:5px solid var(--signal); margin-top:16px; padding:11px 14px; background:var(--paper2); color:#475467; font-size:14px; line-height:1.7; }
.service-strip { display:grid; grid-template-columns:auto 1fr auto auto; align-items:center; gap:12px; border:2px solid var(--ink); background:#fffdf7; padding:12px 15px; margin:4px 0 18px; box-shadow:4px 4px 0 var(--ink); }
.service-dot { width:12px; height:12px; border:2px solid var(--ink); transform:rotate(45deg); background:#f4c64d; }
.service-strip.green .service-dot{background:var(--acid)} .service-strip.red .service-dot{background:var(--signal)}
.service-strip b { font:600 13px var(--mono); letter-spacing:.08em; }
.service-strip small { display:block; color:#596576; font-size:13px; margin-top:2px; }
.service-stat { min-width:72px; padding-left:14px; border-left:1px solid var(--line); text-align:right; }
.workbench { margin-top:18px; gap:18px!important; }
.panel,.workbench>.column { border:2px solid var(--line)!important; border-radius:0!important; background:#fffdf7!important; box-shadow:6px 6px 0 var(--ink)!important; padding:18px!important; }
.panel-title { font:600 13px var(--mono); letter-spacing:.1em; margin-bottom:10px; color:#344054; }
.ask-btn { min-height:54px; border-radius:0!important; border:2px solid var(--ink)!important; background:var(--signal)!important; color:#fff!important; font-family:var(--ui)!important; font-size:16px!important; font-weight:800!important; letter-spacing:.06em!important; box-shadow:4px 4px 0 var(--ink)!important; transition:transform .15s,box-shadow .15s!important; }
.ask-btn:hover { transform:translate(2px,2px); box-shadow:2px 2px 0 var(--ink)!important; }
.answer-box textarea { font-size:22px!important; line-height:1.45!important; font-weight:650!important; background:#101722!important; color:#f8f5ec!important; border-radius:0!important; min-height:220px!important; }
.runtime-note textarea { font:500 12px var(--mono)!important; color:#596576!important; }
.api-note { margin:24px 0 8px; border-top:1px solid var(--line); padding-top:14px; color:#596576; font-size:13px; line-height:1.7; }
.api-note code { font-family:var(--mono); background:#fffdf7; border:1px solid var(--line); padding:2px 5px; }
.chartops-shell>* { animation:rise-in .5s cubic-bezier(.2,.8,.2,1) both; }
.chartops-shell>*:nth-child(2){animation-delay:.05s}.chartops-shell>*:nth-child(3){animation-delay:.1s}.chartops-shell>*:nth-child(4){animation-delay:.15s}
@keyframes rise-in { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
footer { display:none!important; }
@media(max-width:760px){ .gradio-container{width:calc(100% - 20px)!important}.eyebrow{align-items:flex-start;flex-direction:column;gap:7px}.metric-strip{grid-template-columns:1fr 1fr}.metric:nth-child(2){border-right:0}.metric:nth-child(-n+2){border-bottom:1px solid var(--line)}.hero-title{font-size:36px;letter-spacing:-.04em}.deck{font-size:16px}.service-strip{grid-template-columns:auto 1fr}.service-stat{display:none}.panel,.workbench>.column{box-shadow:4px 4px 0 var(--ink)!important} }
@media(prefers-reduced-motion:reduce){ .chartops-shell>*{animation:none} }
"""


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_none,
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="CHART/OPS｜Qwen3-VL 圖表分析") as demo:
        gr.HTML(
            """<section class='chartops-shell' lang='zh-TW'>
          <div class='eyebrow'><span>CHART/OPS · QWEN3-VL 08B</span><span>LIVE INFERENCE / GGUF</span></div>
          <h1 class='hero-title'>上傳圖表，<br><em>直接取得答案。</em></h1>
          <p class='deck'>這不是靜態截圖。頁面會把圖表送進本專案 fine-tuned GGUF，透過 llama.cpp 完成真實多模態推論；所有請求受有限佇列、速率限制與輸入大小保護。</p>
          <div class='metric-strip'>
            <div class='metric'><b>Backbone</b><span>Qwen3-VL 8B</span></div>
            <div class='metric'><b>Fine-tuning</b><span>QLoRA · 15K</span></div>
            <div class='metric'><b>Runtime</b><span>GGUF · Q4_K_M</span></div>
            <div class='metric'><b>Admission</b><span>1 active · 6 queued</span></div>
          </div>
          <div class='scope-note'>正式品質數字仍以 AWQ/vLLM 的完整 2,500 題評估為準；本頁 GGUF 服務另有 OOD suite 與長時間 soak report，不跨推論 stack 混用分數。</div>
        </section>"""
        )
        service_status = gr.HTML(render_status())
        with gr.Row(elem_classes=["workbench"]):
            with gr.Column(scale=5, elem_classes=["panel"]):
                gr.HTML("<div class='panel-title'>01 / 圖表來源</div>")
                image_input = gr.Image(type="pil", label="上傳圖表", height=390)
            with gr.Column(scale=5, elem_classes=["panel"]):
                gr.HTML("<div class='panel-title'>02 / 輸入問題</div>")
                question = gr.Textbox(
                    label="問題", placeholder="例如：哪個類別的數值最高？", lines=4
                )
                response_mode = gr.Radio(
                    [SHORT, EXPLAIN], value=SHORT, label="回答模式"
                )
                with gr.Accordion("生成參數", open=False):
                    max_tokens = gr.Slider(
                        8,
                        SETTINGS.max_output_tokens,
                        value=64,
                        step=8,
                        label="Max output tokens",
                    )
                ask = gr.Button(
                    "開始分析圖表 →", variant="primary", elem_classes=["ask-btn"]
                )
        with gr.Row(elem_classes=["workbench"]):
            with gr.Column(elem_classes=["panel"]):
                gr.HTML("<div class='panel-title'>03 / 模型回答</div>")
                output = gr.Textbox(label="回答", lines=7, elem_classes=["answer-box"])
                runtime = gr.Textbox(
                    label="Runtime trace", elem_classes=["runtime-note"]
                )
        gr.HTML(
            """<div class='api-note'><b>Operational endpoints</b> · <code>/healthz</code> liveness · <code>/readyz</code> model readiness · <code>/api/status</code> queue and latency · <code>/metrics</code> Prometheus format</div>"""
        )
        inputs = [image_input, question, response_mode, max_tokens]
        outputs = [output, runtime, service_status]
        ask.click(gradio_answer, inputs, outputs, concurrency_limit=1)
        question.submit(gradio_answer, inputs, outputs, concurrency_limit=1)
        demo.load(render_status, outputs=service_status, show_progress="hidden")
        timer = gr.Timer(value=10, active=True)
        timer.tick(render_status, outputs=service_status, show_progress="hidden")
    return demo


@asynccontextmanager
async def lifespan(_: FastAPI):
    MANAGER.start_background()
    yield
    MANAGER.stop()


api = FastAPI(
    title="CHART/OPS Qwen3-VL",
    version="1.0.0",
    lifespan=lifespan,
)


@api.get("/healthz")
def healthz():
    return {"status": "ok", "uptime_s": METRICS.snapshot()["uptime_s"]}


@api.get("/readyz")
def readyz():
    status = MANAGER.status()
    code = 200 if status["ready"] else 503
    return JSONResponse(status_code=code, content=status)


@api.get("/api/status")
def api_status():
    return status_payload()


@api.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return METRICS.prometheus(MANAGER.state)


@api.post("/api/v1/infer")
def api_infer(
    payload: InferPayload,
    request: Request,
    x_admin_token: str | None = Header(default=None),
):
    try:
        image = decode_data_uri(payload.image_data_uri, SETTINGS.max_image_bytes)
        result = SERVICE.infer(
            image,
            payload.question,
            payload.response_mode,
            payload.max_tokens,
            client_id=_client_id(request),
            bypass_rate_limit=_is_admin(x_admin_token),
        )
        return {
            "answer": result.answer,
            "trace": result.trace,
            "request_id": result.request_id,
            "latency_s": result.latency_s,
        }
    except InputRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_s)},
        ) from exc
    except QueueSaturated as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueWaitTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log_event("api_unhandled_error", error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail="未預期的推論錯誤。") from exc


demo = build_demo()
demo.queue(default_concurrency_limit=1, max_size=SETTINGS.max_queue_size)
app = gr.mount_gradio_app(api, demo, path="/", theme=theme, css=CSS)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        access_log=False,
        timeout_keep_alive=15,
    )
