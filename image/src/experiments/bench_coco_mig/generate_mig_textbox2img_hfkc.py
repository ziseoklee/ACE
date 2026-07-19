"""Compatibility entry point for ACE's former HFKC development name."""

from experiments.bench_coco_mig.generate_mig_textbox2img_ace import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
