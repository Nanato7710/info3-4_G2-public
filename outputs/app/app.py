from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import gradio as gr
import torch
from PIL import Image

from inference import (
    AppConfigurationError,
    InferenceEngine,
    ThresholdConfig,
    UserInputError,
    format_results,
    load_genre_names,
    load_threshold_config,
    select_device,
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
RESULT_HEADERS = ["rank", "genre", "score", "threshold", "判定"]
RESULT_DATATYPES = ["number", "str", "number", "number", "str"]
LOGGER = logging.getLogger(__name__)


def _render_scores(
    scores: Sequence[float] | None,
    thresholds: Sequence[float],
    genre_names: Sequence[str],
) -> tuple[list[list[object]], list[list[object]], list[list[object]], str]:
    if scores is None or len(scores) == 0:
        return [], [], [], "画像を選択して「推論する」を押してください。"
    top_five, candidates, all_rows = format_results(
        scores,
        thresholds,
        genre_names,
    )
    status = f"推論完了。ジャンル別閾値を満たした候補は{len(candidates)}件です。"
    return top_five, candidates, all_rows, status


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return select_device()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise AppConfigurationError("CUDAを利用できません")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise AppConfigurationError("MPSを利用できません")
    return device


def build_demo(
    engine: InferenceEngine,
    threshold_config: ThresholdConfig,
    example_image: Path | None = None,
) -> gr.Blocks:
    genre_names = engine.genre_names
    default_thresholds = threshold_config.thresholds
    initial_scores: list[float] = []
    initial_top_five: list[list[object]] = []
    initial_candidates: list[list[object]] = []
    initial_all_rows: list[list[object]] = []
    initial_status = "画像を選択して「推論する」を押してください。"
    initial_image_path: str | None = None
    if example_image is not None:
        resolved_example = example_image.resolve()
        scores = engine.predict(resolved_example)
        (
            initial_top_five,
            initial_candidates,
            initial_all_rows,
            initial_status,
        ) = _render_scores(scores, default_thresholds, genre_names)
        initial_scores = scores.tolist()
        initial_image_path = str(resolved_example)

    def run_inference(image, *threshold_values):
        try:
            scores = engine.predict(image)
            rendered = _render_scores(scores, threshold_values, genre_names)
            return scores.tolist(), *rendered
        except UserInputError as exc:
            return [], [], [], [], f"エラー: {exc}"
        except Exception as exc:  # Gradio should not expose internal stack traces.
            LOGGER.exception("inference failed")
            return [], [], [], [], f"エラー: {exc}"

    def load_preview(image_path):
        if not image_path:
            return None, "画像を選択して「推論する」を押してください。"
        try:
            with Image.open(image_path) as image:
                return image.convert("RGB"), "画像を読み込みました。"
        except Exception:
            LOGGER.exception("preview failed")
            return None, "エラー: 画像を読み込めません"

    def refresh_results(scores, *threshold_values):
        try:
            return _render_scores(scores, threshold_values, genre_names)
        except Exception as exc:
            LOGGER.exception("threshold update failed")
            return [], [], [], f"エラー: {exc}"

    def reset_thresholds(scores):
        rendered = _render_scores(scores, default_thresholds, genre_names)
        return *default_thresholds, *rendered

    def clear_results():
        return None, [], [], [], [], "画像を選択して「推論する」を押してください。"

    with gr.Blocks(
        title="アニメ画像ジャンル推論",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown("# アニメ画像ジャンル推論")
        gr.Markdown(
            "画像から19ジャンルのscoreを計算します。"
            "候補判定にはseed 44のvalidationで調整したジャンル別閾値を使います。"
        )
        gr.Markdown(
            f"モデル: `{threshold_config.model_id}` / seed: "
            f"`{threshold_config.seed}` / device: `{engine.device}` / "
            f"validation Macro F1: `{threshold_config.validation_macro_f1:.4f}`",
        )

        scores_state = gr.State(value=initial_scores)
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.File(
                    label="入力画像",
                    type="filepath",
                    file_count="single",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                    value=initial_image_path,
                )
                image_preview = gr.Image(
                    label="入力画像のpreview",
                    interactive=False,
                    image_mode="RGB",
                    value=initial_image_path,
                )
                run_button = gr.Button("推論する", variant="primary")
                status = gr.Markdown(initial_status)
            with gr.Column(scale=2):
                top_five = gr.Dataframe(
                    headers=RESULT_HEADERS,
                    datatype=RESULT_DATATYPES,
                    value=initial_top_five,
                    label="score上位5ジャンル",
                    interactive=False,
                )
                candidates = gr.Dataframe(
                    headers=RESULT_HEADERS,
                    datatype=RESULT_DATATYPES,
                    value=initial_candidates,
                    label="ジャンル別閾値を満たした候補",
                    interactive=False,
                )

        all_scores = gr.Dataframe(
            headers=RESULT_HEADERS,
            datatype=RESULT_DATATYPES,
            value=initial_all_rows,
            label="19ジャンルのscore",
            interactive=False,
        )

        with gr.Accordion("ジャンル別閾値", open=False):
            gr.Markdown(
                "各閾値は0.00から1.00まで0.01刻みで変更できます。"
                "変更はこの画面の候補表示だけに反映されます。"
            )
            reset_button = gr.Button("validation既定値へ戻す")
            threshold_sliders = []
            for start in range(0, len(genre_names), 3):
                with gr.Row():
                    for index in range(start, min(start + 3, len(genre_names))):
                        threshold_sliders.append(
                            gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=default_thresholds[index],
                                step=0.01,
                                precision=2,
                                label=genre_names[index],
                            )
                        )

        run_button.click(
            fn=run_inference,
            inputs=[image_input, *threshold_sliders],
            outputs=[scores_state, top_five, candidates, all_scores, status],
            api_name="predict",
        )
        image_input.change(
            fn=load_preview,
            inputs=[image_input],
            outputs=[image_preview, status],
            show_progress="hidden",
        )
        for slider in threshold_sliders:
            slider.release(
                fn=refresh_results,
                inputs=[scores_state, *threshold_sliders],
                outputs=[top_five, candidates, all_scores, status],
                show_progress="hidden",
            )
        reset_button.click(
            fn=reset_thresholds,
            inputs=[scores_state],
            outputs=[
                *threshold_sliders,
                top_five,
                candidates,
                all_scores,
                status,
            ],
            show_progress="hidden",
        )
        image_input.clear(
            fn=clear_results,
            outputs=[
                image_preview,
                scores_state,
                top_five,
                candidates,
                all_scores,
                status,
            ],
            show_progress="hidden",
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Gradio inference app.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/final-tri-model/runs/seed_44/best_model.pth",
    )
    parser.add_argument(
        "--threshold-config",
        type=Path,
        default=APP_DIR / "threshold.json",
    )
    parser.add_argument("--genres", type=Path, default=APP_DIR / "genres.json")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--example-image",
        type=Path,
        help="起動時に推論して表示する確認用画像",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    try:
        genre_names = load_genre_names(args.genres.resolve())
        threshold_config = load_threshold_config(
            args.threshold_config.resolve(),
            genre_names,
        )
        engine = InferenceEngine(
            args.checkpoint.resolve(),
            genre_names,
            threshold_config,
            device=_resolve_device(args.device),
        )
    except AppConfigurationError as exc:
        raise SystemExit(f"起動できません: {exc}") from exc
    demo = build_demo(
        engine,
        threshold_config,
        example_image=args.example_image,
    )
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=False,
    )


if __name__ == "__main__":
    main()
