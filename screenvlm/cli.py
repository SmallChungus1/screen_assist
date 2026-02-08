import argparse
import sys
from .config import settings

def run_command(args):
    print("Starting UI...")
    # TODO: Import and run app
    from .app import main as app_main
    app_main()

def ingest_command(args):
    print(f"Ingesting docs from {args.docs} to {args.persist}...")
    if args.rebuild:
        print("Rebuild flag set.")
    # TODO: Implement ingestion
    from .rag.ingest import ingest_docs
    ingest_docs(args.docs, args.persist, args.rebuild)

def merge_command(args):
    print(f"Merging adapter to {args.out} with dtype {args.dtype}...")
    from .vlm.loader import merge_adapter
    merge_adapter(args.out, args.dtype)

def doctor_command(args):
    print("Running doctor checks...")
    checks = []
    
    # Check 1: Import model deps
    try:
        import torch
        import transformers
        import peft
        checks.append(("Model dependencies", "PASS"))
    except ImportError as e:
        checks.append(("Model dependencies", f"FAIL ({e})"))

    # Check 2: Adapter dir
    import os
    if os.path.exists(settings["adapter_dir"]):
        checks.append(("Adapter directory", "PASS"))
    else:
        checks.append(("Adapter directory", f"FAIL (Not found: {settings['adapter_dir']})"))

    # Check 3: Screen capture
    try:
        import mss
        with mss.mss() as sct:
            sct.shot(mon=-1, output='doctor_test.png')
        if os.path.exists('doctor_test.png'):
            os.remove('doctor_test.png')
            checks.append(("Screen capture", "PASS"))
        else:
            checks.append(("Screen capture", "FAIL (No image produced)"))
    except Exception as e:
        checks.append(("Screen capture", f"FAIL ({e})"))

    # Check 4: Chroma
    try:
        import chromadb
        checks.append(("ChromaDB", "PASS"))
    except ImportError:
         checks.append(("ChromaDB", "FAIL (ImportError)"))

    print("\nHealth Check Results:")
    for name, status in checks:
        print(f"{name:.<30} {status}")

def main():
    parser = argparse.ArgumentParser(description="screenvlm CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    subparsers.add_parser("run", help="Launch the UI app")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents for RAG")
    ingest_parser.add_argument("--docs", default=settings["docs_dir"], help="Path to documents")
    ingest_parser.add_argument("--persist", default=settings["chroma_dir"], help="Path to Chroma DB")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Rebuild index")

    # merge
    merge_parser = subparsers.add_parser("merge", help="Merge adapter into base model")
    merge_parser.add_argument("--out", required=True, help="Output directory")
    merge_parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp16", help="Data type")

    # doctor
    subparsers.add_parser("doctor", help="Check system health")

    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "ingest":
        ingest_command(args)
    elif args.command == "merge":
        merge_command(args)
    elif args.command == "doctor":
        doctor_command(args)

if __name__ == "__main__":
    main()
