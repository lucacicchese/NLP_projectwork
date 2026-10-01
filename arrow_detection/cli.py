"""CLI entry-point for the YOLO arrow detection baseline."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arrow Detection: YOLO baseline")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    p_yolo = sub.add_parser("yolo", help="Run YOLO arrow detection on images")
    p_yolo.add_argument("--input_dir", required=True, help="Directory with input images")
    p_yolo.add_argument("--output_dir", required=True, help="Output directory (csv/ + imgs/)")
    p_yolo.add_argument(
        "--model",
        default="best.pt",
        help="HuggingFace model ID or local .pt path (default: best.pt)",
    )
    p_yolo.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_yolo.add_argument(
        "--arrow_classes",
        nargs="+",
        default=None,
        help="Class names to treat as arrows (default: arrow,arrows,arrowhead,line,connection,connector)",
    )
    p_yolo.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | cuda:0")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "yolo":
        from .yolo_detector import YOLOArrowDetector
        detector = YOLOArrowDetector(
            model_id=args.model,
            device=args.device,
            conf_threshold=args.conf,
            arrow_classes=args.arrow_classes,
        )
        detector.run_detection(input_dir=args.input_dir, output_dir=args.output_dir)

    else:
        parser.print_help()
