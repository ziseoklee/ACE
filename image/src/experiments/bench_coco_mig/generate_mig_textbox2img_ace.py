import inspect
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from baselines import StableDiffusionACEPipelineWrapper
from configs import config  # noqa: F401 - Import to register structured configs with ConfigStore
from configs.config_benchmark import MIGBenchConfig
from configs.config_method import BaseMethodConfig
from experiments.general_utils import seed_everything
from experiments.pipeline_utils import create_pipeline
from utils_layout import draw_layout

GlobalHydra.instance().clear()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    raise EnvironmentError("CUDA device not found. A GPU is required to run this script.")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)-12s - %(levelname)-8s - %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="coco-mig", version_base=None)
def main(cfg: DictConfig) -> None:
    # Convert DictConfig to structured config for better type checking
    bench_config: MIGBenchConfig = cast(MIGBenchConfig, OmegaConf.to_object(cfg))
    method_config = bench_config.method

    # Set efficiency mode if specified
    if hasattr(bench_config, "mode") and bench_config.mode == "efficiency":
        logger.info("Efficiency mode enabled: Using fp32 and disabling torch.compile.")
        method_config.set_efficiency_mode()

    logger.info(f"Running {bench_config.name} with {method_config.name} ({method_config.arch})")

    ######### Prepare pipeline ##########
    pipe = create_pipeline(method_config, device=DEVICE)
    if not isinstance(pipe, StableDiffusionACEPipelineWrapper):
        raise TypeError("gen-mig-box-ace requires an NR, FKC, or ACE method configuration.")
    logger.info(f"Successfully loaded {method_config.name} pipeline")
    #####################################

    run_mig_benchmark(bench_config, method_config, pipe)


def run_mig_benchmark(bench_config: MIGBenchConfig, method_config: BaseMethodConfig, pipe):
    """Runs the MIG Benchmark."""

    # Get unique method directory name to avoid overwriting
    base_output_dir = Path(bench_config.output_base_dir)
    method_dir_name = get_method_dir_name(base_output_dir, method_config, bench_config.seed, bench_config.run_name)

    # Create the method directory
    method_output_dir = base_output_dir / method_dir_name
    method_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Output directory: {method_output_dir}")
    logger.info(f"🌱 Seeds: {bench_config.seed}")
    logger.info(f"🎯 Mode: {bench_config.mode}")
    logger.info(f"🎯 Target levels: {bench_config.target_levels}")

    ########## Prepare dataset ##########
    # Load JSONL dataset
    dataset = []
    with open(bench_config.dataset_jsonl_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(dataset)} samples from {bench_config.dataset_jsonl_path}")
    #####################################

    ########## Generate images ##########
    err_cnt = 0
    latencies = {}
    vram_usages = {}
    samples_processed = 0
    samples_skipped = 0

    for idx, sample in enumerate(tqdm(dataset, desc="COCO-MIG Bench")):
        for itr, seed in enumerate(bench_config.seed):
            # Extract data from sample
            prompt = sample["prompt"]
            tags = parse_tags(sample)
            phrases = sample.get("phrases", [])
            bounding_boxes = sample.get("bounding_boxes", [])
            level = sample.get("level", "N/A")

            # pass if level is not in target_levels
            if level not in bench_config.target_levels and level != "N/A":
                logger.debug(f"Skipping sample {idx}_{itr} with level {level} not in target levels.")
                time.sleep(0.01)  # Small sleep to allow logging to flush and pbar to update
                samples_skipped += 1
                continue

            # Initialize level tracking if not exists
            if level not in latencies:
                latencies[level] = []
                vram_usages[level] = []

            # Check if generated sample already exists
            img_name = f"{idx}_{itr}_{level}_" + prompt
            # Check for any existing particle files for this sample
            existing_files = list(method_output_dir.glob(f"{img_name}_particle_*_weight_*.jpg"))

            if existing_files:
                logger.debug(
                    f"Sample {idx}_{itr} ({img_name}) already exists with {len(existing_files)} particles. Skipping..."
                )
                time.sleep(0.01)  # Small sleep to allow logging to flush and pbar to update
                samples_skipped += 1
                continue

            try:
                # Generate image
                torch.cuda.reset_peak_memory_stats()
                start_time = time.perf_counter()

                # Receive all images and weights from the function
                generated_images, weights = generate_image(
                    pipe,
                    method_config,
                    prompt,
                    tags,
                    phrases,
                    bounding_boxes,
                    seed=seed,
                )

                latencies[level].append(time.perf_counter() - start_time)
                vram_usages[level].append(torch.cuda.max_memory_allocated() / 1024**3)

                # Save the generated image
                # Loop through all returned images and weights to save them
                for i, (img, weight) in enumerate(zip(generated_images, weights)):
                    weight_str = f"{weight.item():.4f}"
                    particle_output_path = (
                        method_output_dir / f"{img_name}_particle_{i:02d}_weight_{weight_str}.jpg"
                    )

                    img.save(particle_output_path)

                    # Save layout image for methods that support it
                    if len(bounding_boxes) > 0 and len(phrases) > 0:
                        boxes = [tuple(bbox) for bbox in bounding_boxes]
                        layout_img = draw_layout(img, phrases=phrases, boxes=boxes)
                        particle_output_layout_path = (
                            method_output_dir / f"{img_name}_particle_{i:02d}_weight_{weight_str}_layout.jpg"
                        )
                        layout_img.save(particle_output_layout_path)

                samples_processed += 1

            except Exception as e:
                logger.error(f"Error generating image for sample {idx}, iteration {itr}: {e}")
                err_cnt += 1
                continue

    logger.info("✅ MIG Benchmark completed:")
    logger.info(f"   📊 Processed: {samples_processed}, Skipped: {samples_skipped}, Errors: {err_cnt}")
    logger.info(f"📁 Generated images saved to: {method_output_dir}")
    #####################################

    ############ Log Results ############
    # Calculate overall metrics
    all_latencies = []
    all_vram_usages = []
    for level_latencies in latencies.values():
        all_latencies.extend(level_latencies)
    for level_vram in vram_usages.values():
        all_vram_usages.extend(level_vram)

    # Log overall metrics
    if all_latencies:
        overall_latency_mean = np.mean(all_latencies)
        overall_latency_std = np.std(all_latencies)
        overall_vram_mean = np.mean(all_vram_usages)
        overall_vram_std = np.std(all_vram_usages)
        logger.info(
            f"📊 Overall - Avg latency: {overall_latency_mean:.4f} ± {overall_latency_std:.4f}s, "
            f"Avg VRAM: {overall_vram_mean:.4f} ± {overall_vram_std:.4f}GB"
        )

    # Log metrics by level
    for level in sorted(latencies.keys()):
        if latencies[level]:
            latency_mean = np.mean(latencies[level])
            latency_std = np.std(latencies[level])
            vram_mean = np.mean(vram_usages[level])
            vram_std = np.std(vram_usages[level])
            logger.info(
                f"📊 Level {level} - Avg latency: {latency_mean:.4f} ± {latency_std:.4f}s, "
                f"Avg VRAM: {vram_mean:.4f} ± {vram_std:.4f}GB"
            )
        else:
            logger.info(f"📊 Level {level} - No data available")
    #####################################

    write_log_file(
        bench_config,
        method_config,
        latencies,
        vram_usages,
        err_cnt,
        samples_processed,
        samples_skipped,
        method_dir_name,
    )


def parse_tags(sample: dict[str, Any]) -> list[str]:
    """
    Parse tags from the input dictionary.
    """
    tags = []
    pattern = r"^expected_obj[1-6]$"  # MIG benchmark supports up to 6 objects
    filtered_keys = [key for key in sample.keys() if re.match(pattern, key)]

    for key in filtered_keys:
        if isinstance(sample[key], str) and len(sample[key]) > 0:
            tags.append(sample[key])

    return tags


def generate_image(
    pipe,
    method_config: BaseMethodConfig,
    prompt: str,
    tags: list[str],
    phrases: list[str],
    bounding_boxes: list[tuple[float, float, float, float]],
    seed: int,
) -> tuple[list[Image.Image], torch.Tensor]:
    """Generate image using the appropriate method"""
    base_kwargs = {
        "prompt": prompt,
        "tags": tags,
        "steps": method_config.steps,
        "cfg": method_config.cfg,
    }

    # Add method-specific parameters
    generation_kwargs: dict[str, Any] = base_kwargs.copy()
    generation_kwargs.update(method_config.generate_kwargs)

    # These methods require phrases and bounding boxes
    accept_boxes = "boxes" in set(inspect.signature(pipe.generate).parameters.keys())
    accept_phrases = "phrases" in set(inspect.signature(pipe.generate).parameters.keys())
    if accept_boxes and accept_phrases:
        boxes = [tuple(bbox) for bbox in bounding_boxes]
        generation_kwargs.update(
            {
                "phrases": phrases,
                "boxes": boxes,
            }
        )

    # Flow matching based models do not support cfgpp. Ignoring cfgpp parameter.
    accept_cfgpp = "cfgpp" in set(inspect.signature(pipe.generate).parameters.keys())
    if accept_cfgpp:
        generation_kwargs.update(
            {
                "cfgpp": method_config.cfgpp,
            }
        )

    # Generate image
    seed_everything(seed)
    generator = torch.Generator(DEVICE).manual_seed(seed)
    result = pipe.generate(generator=generator, **generation_kwargs)
    return result


def get_method_dir_name(
    base_dir: Path,
    method_config: BaseMethodConfig,
    seeds: list[int],
    run_name: str | None,
) -> str:
    """Return a deterministic, non-interactive output directory name.

    Existing samples are resumed rather than overwritten.  Reproduction shell
    scripts pass explicit run names so evaluation never depends on an
    auto-incremented directory suffix.
    """
    raw_name = run_name or f"{method_config.arch}_seed{'_'.join(map(str, seeds))}"
    clean_name = re.sub(r"[^\w\-.]", "_", raw_name)
    candidate_path = base_dir / clean_name
    if candidate_path.exists():
        logger.warning("Resuming %s; existing matching samples will be skipped.", candidate_path)
    return clean_name


def write_log_file(
    bench_config: MIGBenchConfig,
    method_config: BaseMethodConfig,
    latencies: dict[str, list[float]],
    vram_usages: dict[str, list[float]],
    err_cnt: int,
    samples_processed: int,
    samples_skipped: int,
    method_dir_name: str,
):
    """Writes the final log file for the benchmark run."""
    log_path = (
        Path(bench_config.output_base_dir)
        / f"{method_dir_name}_target_levels_{'_'.join(map(str, bench_config.target_levels))}.log"
    )

    with open(log_path, "w") as f:
        f.write("--- Benchmark Configuration ---\n")
        f.write(f"Benchmark: {bench_config.name}\n")
        f.write(f"Method: {method_config.name} ({method_config.arch})\n")
        if hasattr(bench_config, "mode"):
            f.write(f"Mode: {bench_config.mode}\n")
        f.write(f"Seeds: {bench_config.seed}\n")
        f.write(f"Steps: {method_config.steps}\n")
        f.write(f"CFG: {method_config.cfg}\n")
        f.write(f"CFG++: {method_config.cfgpp}\n")
        f.write(f"Generate kwargs: {method_config.generate_kwargs}\n")
        f.write(f"Dataset: {bench_config.dataset_jsonl_path}\n")
        f.write(f"Target levels: {bench_config.target_levels}\n")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                f.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")
        else:
            f.write("No CUDA device found.\n")

        f.write("\n--- Summary Statistics ---\n")
        f.write(f"Samples processed: {samples_processed}\n")
        f.write(f"Samples skipped: {samples_skipped}\n")
        f.write(f"Total errors: {err_cnt}\n")

        # Calculate overall performance metrics across all tasks
        all_latencies = []
        all_vram_usages = []
        for level_latencies in latencies.values():
            all_latencies.extend(level_latencies)
        for level_vram in vram_usages.values():
            all_vram_usages.extend(level_vram)

        f.write("\n--- Performance Metrics ---\n")
        if all_latencies:
            overall_latency_mean = np.mean(all_latencies)
            overall_latency_std = np.std(all_latencies)
            overall_vram_mean = np.mean(all_vram_usages)
            overall_vram_std = np.std(all_vram_usages)
            f.write(f"Overall - Average latency: {overall_latency_mean:.4f} ± {overall_latency_std:.4f} seconds\n")
            f.write(f"Overall - Average peak VRAM usage: {overall_vram_mean:.4f} ± {overall_vram_std:.4f} GB\n")

        f.write("\n--- Performance Metrics by Level ---\n")
        for level in sorted(latencies.keys()):
            if latencies[level]:
                latency_mean = np.mean(latencies[level])
                latency_std = np.std(latencies[level])
                vram_mean = np.mean(vram_usages[level])
                vram_std = np.std(vram_usages[level])
                f.write(f"Level {level} - Average latency: {latency_mean:.4f} ± {latency_std:.4f} seconds\n")
                f.write(f"Level {level} - Average peak VRAM usage: {vram_mean:.4f} ± {vram_std:.4f} GB\n")
            else:
                f.write(f"Level {level} - No data available\n")

    logger.info(f"📄 Log file saved to: {log_path}")
    logger.info(f"🎉 {bench_config.name} completed successfully!")


if __name__ == "__main__":
    main()
