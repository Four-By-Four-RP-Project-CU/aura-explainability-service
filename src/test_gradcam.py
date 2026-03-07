import argparse

from gradcam import test_gradcam


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick Grad-CAM verification using trained checkpoint.")
    parser.add_argument("--image", required=True, help="Absolute path to input image.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="outputs/debug_heatmap.jpg")
    parser.add_argument("--method", default="gradcampp", choices=["gradcam", "gradcampp"])
    parser.add_argument("--smooth", type=int, default=1)
    args = parser.parse_args()

    test_gradcam(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.out,
        method=args.method,
        smooth=args.smooth,
    )


if __name__ == "__main__":
    main()
